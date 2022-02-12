import cv2
import numpy as np


def get_paragraph(raw_result, x_ths=1.5, y_ths=0.07, mode="ltr"):
    # create basic attributes
    box_group = []
    for box in raw_result:
        all_x = [int(coord[0]) for coord in box[0]]
        all_y = [int(coord[1]) for coord in box[0]]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        height = max_y - min_y
        box_group.append(
            [box[1], min_x, max_x, min_y, max_y, height, 0.5 * (min_y + max_y), 0]
        )  # last element indicates group
    # cluster boxes into paragraph
    current_group = 1
    while len([box for box in box_group if box[7] == 0]) > 0:
        box_group0 = [box for box in box_group if box[7] == 0]  # group0 = non-group
        # new group
        if len([box for box in box_group if box[7] == current_group]) == 0:
            box_group0[0][7] = current_group  # assign first box to form new group
        # try to add group
        else:
            current_box_group = [box for box in box_group if box[7] == current_group]
            mean_height = np.mean([box[5] for box in current_box_group])
            min_gx = min([box[1] for box in current_box_group]) - x_ths * mean_height
            max_gx = max([box[2] for box in current_box_group]) + x_ths * mean_height
            min_gy = min([box[3] for box in current_box_group]) - y_ths * mean_height
            max_gy = max([box[4] for box in current_box_group]) + y_ths * mean_height
            add_box = False
            for box in box_group0:
                same_horizontal_level = (min_gx <= box[1] <= max_gx) or (
                    min_gx <= box[2] <= max_gx
                )
                same_vertical_level = (min_gy <= box[3] <= max_gy) or (
                    min_gy <= box[4] <= max_gy
                )
                if same_horizontal_level and same_vertical_level:
                    box[7] = current_group
                    add_box = True
                    break
            # cannot add more box, go to next group
            if add_box == False:
                current_group += 1
    # arrage order in paragraph
    result = []
    for i in set(box[7] for box in box_group):
        current_box_group = [box for box in box_group if box[7] == i]
        mean_height = np.mean([box[5] for box in current_box_group])
        min_gx = min([box[1] for box in current_box_group])
        max_gx = max([box[2] for box in current_box_group])
        min_gy = min([box[3] for box in current_box_group])
        max_gy = max([box[4] for box in current_box_group])

        text = ""
        while len(current_box_group) > 0:
            highest = min([box[6] for box in current_box_group])
            candidates = [
                box for box in current_box_group if box[6] < highest + 0.4 * mean_height
            ]
            # get the far left
            if mode == "ltr":
                most_left = min([box[1] for box in candidates])
                for box in candidates:
                    if box[1] == most_left:
                        best_box = box
            elif mode == "rtl":
                most_right = max([box[2] for box in candidates])
                for box in candidates:
                    if box[2] == most_right:
                        best_box = box
            text += " " + best_box[0]
            current_box_group.remove(best_box)
        box_height = max_gy - min_gy
        result.append(
            [
                [
                    [min_gx, min_gy],
                    [max_gx, min_gy],
                    [max_gx, max_gy],
                    [min_gx, max_gy],
                ],
                text[1:],
                box_height,
                len(text[1:]),
            ]
        )

    return result


def draw_quads(image, pts, color=(255, 0, 0), thickness=1, isClosed=True):
    raw_image = image.copy()
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)

    draw_image = cv2.polylines(raw_image, [pts], isClosed, color, thickness)
    return draw_image


def show_in_order(img, result, raw, output):
    drawed_img = img.copy()
    color1 = (0, 255, 255)
    color2 = (0, 0, 255)

    for i in range(len(raw)):
        pG = np.array(raw[i][0])
        text = np.array(raw[i][1])
        pG = pG.reshape((-1, 1, 2))
        drawed_img = draw_quads(drawed_img, pG, color1, 3)

    for i in range(len(result)):
        pG = np.array(result[i][0])
        text = np.array(result[i][1])
        pG = pG.reshape((-1, 1, 2))
        drawed_img = draw_quads(drawed_img, pG, color2, 3)

    cv2.imwrite(output, drawed_img)
