from typing import Dict, List, Any, Optional, Tuple, cast
from collections.abc import Mapping

import numpy
import torch

from app.helpers.miscellaneous import find_best_target_match


class SequentialDetector:
    """
    Handles sequential face detection, ByteTrack tracking, and Temporal EMA smoothing.
    Decoupled from the VideoProcessor for thread safety and maintainability.

    This class manages its own state for temporal smoothing to ensure that continuous
    video frames are processed reliably without relying on the global UI state.
    """

    def __init__(self, main_window):
        """
        Initializes the SequentialDetector.

        Args:
            main_window: The main application window instance.
        """
        self.main_window = main_window

        # State tracking for sequential frames
        self.last_detected_faces: List[Dict[str, Any]] = []
        self._smoothed_kps: Dict[int, numpy.ndarray] = {}
        self._smoothed_dense_kps: Dict[int, numpy.ndarray] = {}
        self._smoothed_dense_kps_203: Dict[int, numpy.ndarray] = {}

    def reset_state(self):
        """
        Safely clears all temporal smoothing tracking states and resets the
        underlying ByteTrack tracker. Called when seeking or loading new media.
        """
        self.last_detected_faces.clear()
        self._smoothed_kps.clear()
        self._smoothed_dense_kps.clear()
        self._smoothed_dense_kps_203.clear()

        if hasattr(self.main_window, "models_processor") and hasattr(
            self.main_window.models_processor, "face_detectors"
        ):
            self.main_window.models_processor.face_detectors.reset_tracker()

    def run(
        self,
        frame_rgb: numpy.ndarray,
        local_control_for_worker: dict,
        local_params_for_worker: Optional[dict] = None,
        is_master_edit_active: bool = False,
        frame_tensor: Optional[torch.Tensor] = None,
        detector_control_override: Optional[dict] = None,
        frame_number: int = -1,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Executes sequential face detection to guarantee flawless Temporal EMA smoothing
        and ByteTrack tracking.

        This function implements a highly optimized "Track-and-Filter" pipeline:
        1. Fast Detection: Extracts only bounding boxes and 5-point landmarks for all faces in the frame.
        2. Early-Filtering: Uses ArcFace recognition to immediately discard faces that do not match the UI target(s).
        3. Heavy Detection: Computes complex 68-point or 203-point landmarks ONLY for the matched targets, saving massive GPU resources in crowd scenes.

        It also includes a rigorous Sanitization Shield to prevent corrupted arrays (NaNs, Inf, bad shapes)
        from crashing downstream worker threads.

        Args:
            frame_rgb (numpy.ndarray): The input frame as an RGB numpy array.
            local_control_for_worker (dict): Dictionary of global UI control parameters.
            local_params_for_worker (Optional[dict]): Dictionary of face-specific parameters.
            is_master_edit_active (bool): Indicates if the master face editor UI toggle is active.
            frame_tensor (Optional[torch.Tensor]): Optional pre-allocated PyTorch tensor for the frame (saves VRAM reallocation).
            detector_control_override (Optional[dict]): Optional override for detector settings.
            frame_number (int): Current frame number (used for debugging and warning logs).

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                Sanitized arrays containing (bboxes, kpss_5, kpss_68, kpss_203).
        """
        # VR180 requires specialized spherical detection performed within the FrameWorker.
        # Skip the sequential planar detection here.
        if local_control_for_worker.get("VR180ModeEnableToggle", False):
            return None, None, None, None

        # NOTE: No explicit CUDA stream context is used here.
        # Wrapping detection in torch.cuda.stream(current_stream) caused stream corruption:
        # the feeder thread and FrameWorker pool both share the default CUDA stream, so
        # synchronize() in the feeder stalled ALL pending worker GPU ops, causing cascading
        # CUDA illegal-instruction crashes on TensorRT hardware (PR #176 regression).
        # ONNX Runtime manages its own internal streams; no wrapper is needed.
        control = detector_control_override or local_control_for_worker
        use_landmark = control.get("LandmarkDetectToggle", True)
        landmark_mode = control.get("LandmarkDetectModelSelection", "203")
        from_points = control.get("DetectFromPointsToggle", False)

        # Check if advanced UI features are enabled. Features like LivePortrait, FaceEditor,
        # or Makeup strictly require the dense 203-point landmark model to function.
        requires_203 = False
        if local_params_for_worker:
            for face_id, face_params in local_params_for_worker.items():
                if isinstance(face_params, Mapping):
                    is_face_editor_active = is_master_edit_active and face_params.get(
                        "FaceEditorEnableToggle", False
                    )
                    is_expression_active = face_params.get(
                        "FaceExpressionEnableBothToggle", False
                    )
                    is_auto_mouth_active = face_params.get(
                        "AutoMouthExpressionEnableToggle", False
                    )
                    is_makeup_active = is_master_edit_active and (
                        face_params.get("FaceMakeupEnableToggle", False)
                        or face_params.get("HairMakeupEnableToggle", False)
                        or face_params.get("EyeBrowsMakeupEnableToggle", False)
                        or face_params.get("LipsMakeupEnableToggle", False)
                    )

                    if (
                        is_face_editor_active
                        or is_expression_active
                        or is_auto_mouth_active
                        or is_makeup_active
                    ):
                        requires_203 = True
                        break

        # Handle the PyTorch tensor allocation
        device = self.main_window.models_processor.device
        owns_frame_tensor = frame_tensor is None
        if frame_tensor is None:
            frame_tensor = (
                torch.from_numpy(frame_rgb)
                .to(device, non_blocking=True)
                .permute(2, 0, 1)  # Convert [H, W, C] -> [C, H, W]
            )

        # --- STEP 1: FAST DETECTION (BBoxes and 5-point landmarks ONLY) ---
        # CRITICAL OPTIMIZATION: Heavy landmark detection (68/203 points) is disabled here
        # to rapidly scan the crowd without burning GPU cycles.
        bboxes, kpss_5, _ = self.main_window.models_processor.run_detect(
            frame_tensor,
            control.get("DetectorModelSelection", "RetinaFace"),
            max_num=int(control.get("MaxFacesToDetectSlider", 20)),
            score=control.get("DetectorScoreSlider", 50) / 100.0,
            input_size=(512, 512),
            use_landmark_detection=False,
            from_points=from_points,
            rotation_angles=[0]
            if not control.get("AutoRotationToggle", False)
            else [0, 90, 180, 270],
            use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
            control_override=control,
            bypass_bytetrack=False,
        )

        # --- STEP 2: SMART FILTERING (ArcFace Recognition) ---
        valid_indices: list = []

        # Safely copy the target faces dictionary to avoid concurrent access issues from the UI thread.
        try:
            target_faces = dict(self.main_window.target_faces)
        except Exception:
            target_faces = {}

        rec_model = str(control.get("RecognitionModelSelection", "ArcFace"))
        default_params = dict(self.main_window.default_parameters.data)

        # 2a. Fast-path: skip ArcFace identity verification when the scene contains exactly
        # one detected face and the user has configured exactly one target face. This saves
        # one ArcFace inference per frame at ~5-10 ms each — significant headroom on webcam
        # at 30 fps. The FrameWorker still re-verifies identity per face before swapping,
        # so a wrong-identity slip-through is corrected downstream.
        # Toggle name FastPathSingleFaceToggle (default True). Power users can disable.
        fast_path_enabled = bool(control.get("FastPathSingleFaceToggle", True))
        used_fast_path = False
        if (
            fast_path_enabled
            and isinstance(bboxes, numpy.ndarray)
            and bboxes.shape[0] == 1
            and len(target_faces) == 1
        ):
            valid_indices = [0]
            used_fast_path = True

        # 2b. Strict ArcFace pre-filter (multi-face or multi-target case).
        if (
            not used_fast_path
            and isinstance(bboxes, numpy.ndarray)
            and bboxes.shape[0] > 0
            and len(target_faces) > 0
        ):
            for i in range(len(kpss_5)):
                # Ultra-fast recognition pass using only the 5 basic keypoints
                face_emb, _ = self.main_window.models_processor.run_recognize_direct(
                    frame_tensor, kpss_5[i], "Auto", rec_model
                )

                # Verify if the detected face matches any of our targets
                match, _, _ = find_best_target_match(
                    face_emb,
                    self.main_window.models_processor,
                    target_faces,
                    local_params_for_worker or {},
                    default_params,
                    rec_model,
                )

                # Keep the index only if it's a target face
                if match is not None:
                    valid_indices.append(i)
        elif (
            not used_fast_path
            and isinstance(bboxes, numpy.ndarray)
            and bboxes.shape[0] > 0
        ):
            # If no specific targets are configured (e.g., pure FaceTracking mode), keep everyone
            valid_indices = list(range(len(bboxes)))

        # 2c. Relaxed-threshold ArcFace pass — runs only when the strict pass produced no
        # matches. ArcFace embeddings degrade on side angles, motion blur, and partial
        # occlusion: the strict pass would discard the actual target and produce flashing
        # artefacts. We re-test with the per-target SimilarityThresholdSlider halved (and
        # floored at 20). The FrameWorker performs its own per-face similarity check before
        # swapping, so a relaxed match that turns out to be wrong is filtered downstream.
        if (
            not used_fast_path
            and len(valid_indices) == 0
            and isinstance(bboxes, numpy.ndarray)
            and bboxes.shape[0] > 0
            and len(target_faces) > 0
        ):
            relaxed_local_params: Dict[str, Any] = {}
            if local_params_for_worker:
                for _tgt_id, _per in local_params_for_worker.items():
                    if isinstance(_per, Mapping):
                        _per_copy = dict(_per)
                        if "SimilarityThresholdSlider" in _per_copy:
                            try:
                                _orig = float(_per_copy["SimilarityThresholdSlider"])
                                _per_copy["SimilarityThresholdSlider"] = max(
                                    20.0, _orig * 0.5
                                )
                            except (TypeError, ValueError):
                                pass
                        relaxed_local_params[_tgt_id] = _per_copy
                    else:
                        relaxed_local_params[_tgt_id] = _per
            relaxed_default = dict(default_params)
            if "SimilarityThresholdSlider" in relaxed_default:
                try:
                    _orig = float(relaxed_default["SimilarityThresholdSlider"])
                    relaxed_default["SimilarityThresholdSlider"] = max(
                        20.0, _orig * 0.5
                    )
                except (TypeError, ValueError):
                    pass

            for i in range(len(kpss_5)):
                face_emb, _ = self.main_window.models_processor.run_recognize_direct(
                    frame_tensor, kpss_5[i], "Auto", rec_model
                )
                match, _, _ = find_best_target_match(
                    face_emb,
                    self.main_window.models_processor,
                    target_faces,
                    relaxed_local_params,
                    relaxed_default,
                    rec_model,
                )
                if match is not None:
                    valid_indices.append(i)

        # 2d. Last-resort single-face fallback: if both passes filtered everything out
        # but only one face was detected, allow it through. This catches the rare case
        # where ArcFace produces a near-zero similarity (severe motion blur, occlusion).
        # The FrameWorker still performs identity verification before swapping.
        if (
            len(valid_indices) == 0
            and isinstance(bboxes, numpy.ndarray)
            and bboxes.shape[0] == 1
            and len(target_faces) > 0
        ):
            valid_indices = [0]

        # Apply the filter to eliminate background extras
        filtered_bboxes = (
            bboxes[valid_indices]
            if valid_indices
            else numpy.empty((0, 4), dtype=numpy.float32)
        )
        filtered_kpss_5 = (
            kpss_5[valid_indices]
            if valid_indices
            else numpy.empty((0, 5, 2), dtype=numpy.float32)
        )

        # --- STEP 3: HEAVY LANDMARK DETECTION (On target faces ONLY) ---
        filtered_kpss = []
        filtered_kpss_203 = []

        for i in range(len(filtered_bboxes)):
            current_bbox = filtered_bboxes[i]
            current_kps5 = filtered_kpss_5[i]

            # CRITICAL FALLBACK: Instead of appending None on failure, we use valid arrays
            # to prevent downstream ".copy()" calls from throwing AttributeError.
            kps_standard = current_kps5.copy()

            # FW-LOGIC-FIX 1: Extract forced 203 landmarks FIRST if advanced editing features demand it.
            # This ensures we always have a properly aligned 203 if required, independently of UI settings.
            kps_203_local = numpy.zeros((203, 2), dtype=numpy.float32)
            has_valid_203 = False

            if requires_203:
                lm_203_5, lm_203, _ = (
                    self.main_window.models_processor.run_detect_landmark(
                        frame_tensor,
                        current_bbox,
                        current_kps5,
                        detect_mode="203",
                        score=0.5,
                        use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
                        from_points=True,  # STRICTLY REQUIRED FOR INTERNAL ALIGNMENT
                    )
                )
                if len(lm_203) > 0:
                    kps_203_local = lm_203
                    has_valid_203 = True

                    # If 203 was extracted for fallback, but the user ALSO selected 203
                    # as their primary UI model, update the Swapper's 5 points immediately.
                    if landmark_mode == "203" and len(lm_203_5) > 0:
                        filtered_kpss_5[i] = lm_203_5

                filtered_kpss_203.append(kps_203_local)

            # FW-LOGIC-FIX 2: Extract standard dense landmarks (68, 203 or 478 depending on UI selection)
            if use_landmark:
                # OPTIMIZATION: Reuse the 203 landmarks computed above ONLY IF
                # the user explicitly enabled 'from_points' in the UI.
                # This effectively prevents a redundant and costly neural network forward pass.
                if (
                    landmark_mode == "203"
                    and requires_203
                    and has_valid_203
                    and from_points
                ):
                    kps_standard = kps_203_local.copy()
                    # (filtered_kpss_5[i] is already updated in the block above)
                else:
                    lm_std_5, lm_kpss, _ = (
                        self.main_window.models_processor.run_detect_landmark(
                            frame_tensor,
                            current_bbox,
                            current_kps5,
                            detect_mode=landmark_mode,
                            score=control.get("LandmarkDetectScoreSlider", 50) / 100.0,
                            use_mean_eyes=control.get("LandmarkMeanEyesToggle", False),
                            from_points=from_points,
                        )
                    )
                    if len(lm_kpss) > 0:
                        kps_standard = lm_kpss
                        # Sync the 5-point array with the refined outputs
                        if len(lm_std_5) > 0:
                            filtered_kpss_5[i] = lm_std_5

            filtered_kpss.append(kps_standard)

        # Reformat output arrays to match the expected pipeline signature (CRITICAL TYPE CASTING)
        bboxes = numpy.array(filtered_bboxes, dtype=numpy.float32)
        kpss_5 = numpy.array(filtered_kpss_5, dtype=numpy.float32)

        if len(filtered_kpss) > 0:
            kpss = numpy.array(filtered_kpss, dtype=object)
        else:
            kpss = numpy.empty((0, 5, 2), dtype=numpy.float32)

        if requires_203 and len(filtered_kpss_203) > 0:
            kpss_203 = numpy.array(filtered_kpss_203, dtype=object)
        else:
            kpss_203 = numpy.empty((0, 203, 2), dtype=numpy.float32)

        # Cleanup VRAM
        if owns_frame_tensor:
            del frame_tensor

        # Safely copy arrays before returning to prevent memory corruption when the
        # arrays are accessed by independent worker threads.
        if isinstance(bboxes, numpy.ndarray):
            bboxes = bboxes.copy()
        if isinstance(kpss_5, numpy.ndarray):
            kpss_5 = kpss_5.copy()
        if isinstance(kpss, numpy.ndarray):
            kpss = kpss.copy()
        if isinstance(kpss_203, numpy.ndarray):
            kpss_203 = kpss_203.copy()

        # --- SANITIZATION SHIELD ---
        # Ensures only perfectly formatted data passes through to the FrameWorker.
        # Defends against dimension mismatches, NaNs, and infinite values.
        if isinstance(bboxes, numpy.ndarray):
            if bboxes.dtype == object:
                try:
                    bboxes = bboxes.astype(numpy.float32)
                except Exception:
                    bboxes = numpy.empty((0, 4), dtype=numpy.float32)

            if bboxes.size > 0 and bboxes.ndim == 2 and bboxes.shape[1] == 4:
                valid_mask = numpy.isfinite(bboxes).all(axis=1)
                if not valid_mask.all():
                    bboxes = bboxes[valid_mask]
                    if isinstance(kpss_5, numpy.ndarray) and kpss_5.shape[0] == len(
                        valid_mask
                    ):
                        kpss_5 = kpss_5[valid_mask]
                    if isinstance(kpss, numpy.ndarray) and kpss.shape[0] == len(
                        valid_mask
                    ):
                        kpss = kpss[valid_mask]
                    if isinstance(kpss_203, numpy.ndarray) and kpss_203.shape[0] == len(
                        valid_mask
                    ):
                        kpss_203 = kpss_203[valid_mask]
            else:
                bboxes = numpy.empty((0, 4), dtype=numpy.float32)
        else:
            bboxes = numpy.empty((0, 4), dtype=numpy.float32)

        if bboxes.shape[0] == 0:
            if isinstance(kpss_5, numpy.ndarray):
                kpss_5 = numpy.empty((0, 5, 2), dtype=numpy.float32)
            if isinstance(kpss, numpy.ndarray):
                kpss = numpy.empty((0, 68, 2), dtype=numpy.float32)
            if isinstance(kpss_203, numpy.ndarray):
                kpss_203 = numpy.empty((0, 203, 2), dtype=numpy.float32)

        # Update global tracker state for UI bounding box rendering
        detected_for_state = []
        if bboxes.shape[0] > 0:
            for i in range(len(bboxes)):
                detected_for_state.append({"bbox": bboxes[i], "score": 1.0})
        self.last_detected_faces = detected_for_state

        is_smoothing_enabled = local_control_for_worker.get(
            "KPSSmoothingEnableToggle", True
        )

        # --- TEMPORAL EMA SMOOTHING ---
        # Matches faces between current frame and previous frame based on spatial proximity (centroids).
        # Smooths the keypoints using an Exponential Moving Average to eliminate micro-jitter.
        if is_smoothing_enabled:
            img_h_for_kps, img_w_for_kps = frame_rgb.shape[0], frame_rgb.shape[1]

            if isinstance(kpss_5, numpy.ndarray) and kpss_5.shape[0] > 0:
                kpss_5 = kpss_5.copy()
                n_faces = kpss_5.shape[0]

                new_smoothed_kps = {}
                new_smoothed_dense_kps = {}
                new_smoothed_dense_kps_203 = {}

                valid_kpss = cast(numpy.ndarray, kpss)
                valid_kpss_203 = cast(numpy.ndarray, kpss_203)

                dense_kps_count = (
                    int(kpss.shape[0]) if isinstance(kpss, numpy.ndarray) else 0
                )
                has_dense_kps = dense_kps_count > 0
                if has_dense_kps:
                    valid_kpss = valid_kpss.copy()
                    kpss = valid_kpss

                dense_kps_203_count = (
                    int(kpss_203.shape[0]) if isinstance(kpss_203, numpy.ndarray) else 0
                )
                has_dense_kps_203 = dense_kps_203_count > 0
                if has_dense_kps_203:
                    valid_kpss_203 = valid_kpss_203.copy()
                    kpss_203 = valid_kpss_203

                # Log warnings if array lengths misalign (guards against indexing errors below)
                if has_dense_kps and dense_kps_count != n_faces:
                    print(
                        f"[WARN] Dense KPS count mismatch on frame {frame_number}: kpss_5={n_faces}, dense_kps={dense_kps_count}. Skipping dense smoothing for missing faces."
                    )
                if has_dense_kps_203 and dense_kps_203_count != n_faces:
                    print(
                        f"[WARN] Dense KPS_203 count mismatch on frame {frame_number}: kpss_5={n_faces}, dense_kps_203={dense_kps_203_count}. Skipping dense 203 smoothing for missing faces."
                    )

                for _i in range(n_faces):
                    _raw = kpss_5[_i]
                    dense_kps_available = has_dense_kps and _i < dense_kps_count
                    dense_kps_203_available = (
                        has_dense_kps_203 and _i < dense_kps_203_count
                    )

                    # Skip smoothing if data is malformed or out of image bounds
                    if (
                        _raw is None
                        or _raw.size == 0
                        or numpy.any(numpy.isnan(_raw))
                        or numpy.any(numpy.isinf(_raw))
                    ):
                        continue
                    if (
                        numpy.any(_raw[:, 0] < 0)
                        or numpy.any(_raw[:, 0] >= img_w_for_kps)
                        or numpy.any(_raw[:, 1] < 0)
                        or numpy.any(_raw[:, 1] >= img_h_for_kps)
                    ):
                        continue

                    _centroid_raw = numpy.mean(_raw, axis=0)
                    _best_match_key = None
                    _min_dist = float("inf")

                    # Calculate a dynamic tolerance threshold based on face size
                    _face_w = bboxes[_i][2] - bboxes[_i][0]
                    _face_h = bboxes[_i][3] - bboxes[_i][1]
                    _adaptive_threshold = max(30.0, max(_face_w, _face_h) * 0.4)

                    # Find the corresponding face in the previous frame
                    for _k, _prev_kps in self._smoothed_kps.items():
                        _centroid_prev = numpy.mean(_prev_kps, axis=0)
                        _dist = numpy.linalg.norm(_centroid_raw - _centroid_prev)

                        if _dist < _adaptive_threshold and _dist < _min_dist:
                            _min_dist = float(_dist)
                            _best_match_key = _k

                    # If a match is found, apply Exponential Moving Average (EMA)
                    if _best_match_key is not None:
                        base_alpha = (
                            local_control_for_worker.get("KPSEmaAlphaSlider", 35)
                            / 100.0
                        )

                        # Dynamic Alpha: Increase responsiveness (less smoothing) during fast movement
                        movement_factor = min(1.0, _min_dist / 15.0)
                        dynamic_alpha = base_alpha + movement_factor * (
                            1.0 - base_alpha
                        )

                        new_smoothed_kps[_i] = (
                            dynamic_alpha * _raw
                            + (1.0 - dynamic_alpha)
                            * self._smoothed_kps[_best_match_key]
                        )
                        del self._smoothed_kps[_best_match_key]

                        if dense_kps_available:
                            if _best_match_key in self._smoothed_dense_kps:
                                new_smoothed_dense_kps[_i] = (
                                    dynamic_alpha * valid_kpss[_i]
                                    + (1.0 - dynamic_alpha)
                                    * self._smoothed_dense_kps[_best_match_key]
                                )
                                del self._smoothed_dense_kps[_best_match_key]
                            else:
                                new_smoothed_dense_kps[_i] = valid_kpss[_i].copy()

                        if has_dense_kps_203:
                            if dense_kps_203_available:
                                is_valid_203 = not numpy.all(valid_kpss_203[_i] == 0)
                                if is_valid_203:
                                    if _best_match_key in self._smoothed_dense_kps_203:
                                        new_smoothed_dense_kps_203[_i] = (
                                            dynamic_alpha * valid_kpss_203[_i]
                                            + (1.0 - dynamic_alpha)
                                            * self._smoothed_dense_kps_203[
                                                _best_match_key
                                            ]
                                        )
                                        del self._smoothed_dense_kps_203[
                                            _best_match_key
                                        ]
                                    else:
                                        new_smoothed_dense_kps_203[_i] = valid_kpss_203[
                                            _i
                                        ].copy()
                                else:
                                    # Fallback: Carry over previous 203 points if current ones are zeroes
                                    if _best_match_key in self._smoothed_dense_kps_203:
                                        new_smoothed_dense_kps_203[_i] = (
                                            self._smoothed_dense_kps_203[
                                                _best_match_key
                                            ].copy()
                                        )
                                        valid_kpss_203[_i] = new_smoothed_dense_kps_203[
                                            _i
                                        ]

                    # No previous face found (new face entering frame), initialize state
                    else:
                        new_smoothed_kps[_i] = _raw.copy()
                        if dense_kps_available:
                            new_smoothed_dense_kps[_i] = valid_kpss[_i].copy()
                        if has_dense_kps_203:
                            if dense_kps_203_available:
                                new_smoothed_dense_kps_203[_i] = valid_kpss_203[
                                    _i
                                ].copy()

                    # Write back the smoothed coordinates to the result arrays
                    kpss_5[_i] = new_smoothed_kps[_i]
                    if dense_kps_available:
                        valid_kpss[_i] = new_smoothed_dense_kps[_i]
                    if has_dense_kps_203:
                        if dense_kps_203_available:
                            valid_kpss_203[_i] = new_smoothed_dense_kps_203[_i]

                # Update state for the next frame
                self._smoothed_kps = new_smoothed_kps
                self._smoothed_dense_kps = new_smoothed_dense_kps
                self._smoothed_dense_kps_203 = new_smoothed_dense_kps_203

        # Clear state if smoothing is disabled to prevent stale data buildup
        else:
            self._smoothed_kps.clear()
            self._smoothed_dense_kps.clear()
            self._smoothed_dense_kps_203.clear()

        return bboxes, kpss_5, kpss, kpss_203
