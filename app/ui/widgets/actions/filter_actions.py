from typing import TYPE_CHECKING

from PySide6 import QtWidgets, QtCore

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def filter_target_videos(main_window: "MainWindow", search_text: str = ""):
    main_window.target_videos_filter_worker.stop_thread()
    main_window.target_videos_filter_worker.search_text = search_text
    main_window.target_videos_filter_worker.start()


def filter_input_faces(main_window: "MainWindow", search_text: str = ""):
    main_window.input_faces_filter_worker.stop_thread()
    main_window.input_faces_filter_worker.search_text = search_text
    main_window.input_faces_filter_worker.start()


def select_all_input_faces(main_window: "MainWindow"):
    """全选输入人脸列表中的所有项目"""
    if not main_window.cur_selected_target_face_button:
        return
    
    cur_selected_target_face_button = main_window.cur_selected_target_face_button
    
    for i in range(main_window.inputFacesList.count()):
        item = main_window.inputFacesList.item(i)
        if item and not item.isHidden():
            button = main_window.inputFacesList.itemWidget(item)
            if button and hasattr(button, 'setChecked') and hasattr(button, 'face_id'):
                button.setChecked(True)
                cur_selected_target_face_button.assigned_input_faces[button.face_id] = button.embedding_store
    
    cur_selected_target_face_button.calculate_assigned_input_embedding()
    from app.ui.widgets.actions import common_actions as common_widget_actions
    common_widget_actions.refresh_frame(main_window)


def filter_merged_embeddings(main_window: "MainWindow", search_text: str = ""):
    main_window.merged_embeddings_filter_worker.stop_thread()
    main_window.merged_embeddings_filter_worker.search_text = search_text
    main_window.merged_embeddings_filter_worker.start()


def update_filtered_list(
    main_window: "MainWindow",
    filter_list_widget: QtWidgets.QListWidget,
    visible_indices: list,
):
    for i in range(filter_list_widget.count()):
        filter_list_widget.item(i).setHidden(True)

    # Show only the items in the visible_indices list
    for i in visible_indices:
        filter_list_widget.item(i).setHidden(False)
