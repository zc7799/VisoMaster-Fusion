# Quick Start Guide

> Your first face swap, step by step.

---

## Before You Begin

**What you will need:**

- VisoMaster Fusion installed and running
- A target video or image — the footage you want to swap faces in
- One or more source images — clear, well-lit photos of the face you want to swap in (front-facing, no glasses, no heavy shadows)
- A reasonably capable GPU — Nvidia with 8 GB+ VRAM is recommended

> **Tip:** More source images = better results. Three to ten varied photos of the same person (different angles, expressions, lighting) will produce a more robust and consistent swap than a single image.

---

## Step 1 — Load Your Target Media

1. Click the **open file** button in the top-left toolbar and select your target video or image
2. The video will appear in the centre preview
3. Scrub through it to confirm it loaded correctly

> Use **`C`** and **`V`** on your keyboard to step through the video one frame at a time.

---

## Step 2 — Load Your Source Face(s)

1. In the left panel, click **Add Face** and select your source image(s)
2. Each image becomes a face card — VisoMaster will auto-detect the face in it
3. Your face card is now ready to be assigned to a face in the target

### Using multiple source images (Embeddings)

If you have three or more photos of the same person, combine them into a single embedding for a more accurate and consistent result:

1. Hold **Ctrl** and click to select multiple face cards
2. **Right-click** and choose **Create Embedding**
3. The embedding replaces the individual cards and represents a combined identity

Aim for 3–10 varied images.

---

## Step 3 — Detect and Assign Faces

1. Click **Detect Faces** — VisoMaster scans the current frame and draws boxes around each detected face
2. Click the face box in the preview that you want to swap
3. Click your source face card to assign it — the card and the face box are now linked
4. Repeat for any additional faces, assigning a different card to each

> **Tip:** If you only need to swap one person in a scene with multiple faces, right-click the other detected faces and remove them. Fewer active faces means faster processing and fewer conflicts.

---

## Step 4 — Use Markers for Changing Scenes

If your video has multiple scenes or the settings need to change partway through, markers let you apply different settings at different points in the timeline.

- Navigate to the frame where you want a settings change
- Adjust your settings
- Click **Add Marker** — it stores the current settings at that timecode
- Settings stay active until the next marker; the last marker applies to the end of the video
- Use the **Previous / Next Marker** buttons to jump between them quickly

---

## Step 5 — Record

1. Once you are happy with the preview, click **Record**
2. VisoMaster processes every frame and saves the output
3. The finished file lands in your `outputs` folder

---

## Performance Tips

- Set the backend to **TensorRT-Engine** for best performance on Nvidia GPUs — on first activation, models take a few minutes to compile, this is normal
- Set **Thread Count to 1** if you run into VRAM issues or crashes during recording
- After changing settings, click **Clear VRAM** and re-enable face swap before recording

---

## Going Further

For a full explanation of every setting and what it does, see the [User Manual](./user_manual.md).

For workflow tips, advanced settings, and community presets, join the [Discord](https://discord.gg/5rx4SQuDbp).

---

## Optional Post-Processing

A common community workflow for video is to halve the frame rate before swapping to speed up processing, then restore it afterward using AI frame interpolation — [Topaz Video AI](https://www.topazlabs.com/topaz-video-ai) or [FlowFrames](https://www.fcportables.com/flowframes-portable/) both work well. This is entirely optional.
