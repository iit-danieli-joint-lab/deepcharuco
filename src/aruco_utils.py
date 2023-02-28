import numpy as np
import cv2


def get_aruco_dict(board_name):
    return cv2.aruco.Dictionary_get(getattr(cv2.aruco, board_name))


def board_image(board, resolution: tuple[int, int],
                row_count: int, col_count: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates board image of given resolution and returns the inner corners pixel position
    Resolution (width, height)
    """
    img = cv2.cvtColor(board.draw(outSize=resolution), cv2.COLOR_GRAY2BGR)
    pixel_offset = np.array([resolution[0] / col_count, resolution[1] / row_count])

    # (row_id, col_idx, (x, y) pixel coords)
    inn_rc = np.arange(1, row_count)
    inn_cc = np.arange(1, col_count)
    corners = np.array(np.meshgrid(inn_rc, inn_cc)).reshape((2, -1)).T * pixel_offset
    return img, corners.astype(int)


def draw_inner_corners(img: np.ndarray, corners: np.ndarray,
                       draw_ids=False, radius=2) -> np.ndarray:
    assert img.ndim == 3 and img.shape[-1] == 3
    img = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1

    for ith, corner in enumerate(corners):
        corner = np.round(corner).astype(np.int)

        if corner[0] > img.shape[1] or corner[1] > img.shape[0]:
            continue
        cv2.circle(img, tuple(corner), radius=radius, color=(0, 0, 255),
                   thickness=text_thickness)

        if draw_ids:
            label_size, _ = cv2.getTextSize(str(ith), font, .5, text_thickness)
            pos = (corner[0] - label_size[0] // 2, corner[1] + label_size[1] // 2 - 10)
            cv2.putText(img, str(ith), pos, font, .3, (0, 0, 255), text_thickness)
    return img


def pred_to_keypoints(loc_hat: np.ndarray, ids_hat: np.ndarray, dust_bin_ids: int):
    # Steps to convert model output to keypoints:
    # Do an argmax of ids_hat, care where != bin
    # For each of those regions get loc_hat infos
    # Use ids coords + argmax on loc_hat infos to select pixel
    # TODO
    print(ids_hat.shape)
    np.argmax(ids_hat, axis=-1)


def draw_circle_pred(img: np.ndarray, loc: np.ndarray, ids: np.ndarray,
                     dust_bin_ids: int, draw_ids=False, radius=2):
    # TODO: This assumes to have class_indices for keypoints already
    # TODO: Conversion for actual prediction

    # Find in which regions the corners are
    roi = np.argwhere(ids != dust_bin_ids)
    ids_found = ids[ids != dust_bin_ids]

    region_pixel = loc[ids != dust_bin_ids]
    
    # Recover exact pixel in original resolution
    xs = 8 * roi[:, 1] + (region_pixel % 8)
    ys = 8 * roi[:, 0] + (region_pixel / 8).astype(int)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    for (*corner, ith) in zip(xs, ys, ids_found):
        cv2.circle(img, corner, radius=radius, color=(255, 0, 0),
                   thickness=text_thickness)

        if draw_ids:
            label_size, _ = cv2.getTextSize(str(ith), font, .5, text_thickness)
            pos = (corner[0] - label_size[0] // 2, corner[1] + label_size[1] // 2 + 5)
            cv2.putText(img, str(ith), pos, font, .3, (255, 0, 0), text_thickness)
    return img


def get_board(configs):
    board = cv2.aruco.CharucoBoard_create(
        squaresX=configs.col_count,
        squaresY=configs.row_count,
        squareLength=configs.square_len,
        markerLength=configs.marker_len,
        dictionary=get_aruco_dict(configs.board_name))
    return board
