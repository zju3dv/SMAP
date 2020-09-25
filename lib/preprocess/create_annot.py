from pycocotools.coco import COCO
import numpy as np
import json
import os

root_dir = 'data/coco2017'
data_type = 'train2017'
anno_name = 'person_keypoints_{}.json'.format(data_type)
anno_file = os.path.join(root_dir, 'annotations', anno_name)
output_json_file = os.path.join(root_dir, 'coco_keypoints_{}.json'.format(data_type))
coco_kps = COCO(anno_file)

catIds = coco_kps.getCatIds(catNms=['person'])
imgIds = coco_kps.getImgIds(catIds=catIds)

COCO2CMUP = [-1, -1, -1, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]

def main():
    output_json = dict()
    output_json['root'] = []
    count = 0
    min_width = 1000
    min_height = 1000
    for i in range(len(imgIds)):
        bodys = list()
        img = coco_kps.loadImgs(imgIds[i])[0]
        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
        annos = coco_kps.loadAnns(annIds)
        for anno in annos:
            if anno['num_keypoints'] < 3:
                continue
            body = np.asarray(anno['keypoints'])
            body.resize((17, 3))
            body_new = np.zeros((15, 11))
            for k in range(len(COCO2CMUP)):
                if COCO2CMUP[k] < 0:
                    continue
                body_new[k][0] = body[COCO2CMUP[k]][0]
                body_new[k][1] = body[COCO2CMUP[k]][1]
                body_new[k][3] = body[COCO2CMUP[k]][2]  # Z(body_new[k][2]) is preserved as 0
            middle_shoulder = (body[5] + body[6]) / 2
            middle_hip = (body[11] + body[12]) / 2
            # hip
            body_new[2][0] = middle_hip[0]
            body_new[2][1] = middle_hip[1]
            body_new[2][3] = min(body[11][2], body[12][2])
            # neck
            body_new[0][0] = (middle_shoulder[0] - middle_hip[0])*0.185 + middle_shoulder[0]
            body_new[0][1] = (middle_shoulder[1] - middle_hip[1])*0.185 + middle_shoulder[1]
            body_new[0][3] = min(body_new[2][3], body[5][2], body[6][2])
            #head top
            """
            body_new[1][0] = (body[0][0] - body_new[0][0]) + body[0][0]
            body_new[1][1] = (body[0][1] - body_new[0][1]) + body[0][1]
            body_new[1][3] = min(body[0][2], body_new[0][3])
            """
            body_new[:, 7] = img['width']  # fx
            body_new[:, 8] = img['width']  # fy
            body_new[:, 9] = img['width'] / 2  # cx
            body_new[:, 10] = img['height'] / 2 # cy
            bodys.append(body_new.tolist())
        if len(bodys) < 1:
            continue
        this_pic = dict()
        this_pic["dataset"] = "COCO"
        this_pic["img_paths"] = data_type + "/" + img['file_name']
        this_pic["img_width"] = img['width']
        this_pic["img_height"] = img['height']
        this_pic["image_id"] = img['id']
        this_pic["cam_id"] = 0
        this_pic["bodys"] = bodys
        this_pic["isValidation"] = 0  # ATTN !
        output_json["root"].append(this_pic)
        count += 1
        min_width = min(min_width, img['width'])
        min_height = min(min_height, img['height'])
        print("writed {}".format(img['file_name']))

    with open(output_json_file, 'w') as f:
        json.dump(output_json, f)
    print("Generated {} annotations, min width is {}, min height is {}.".format(count, min_width, min_height))


if __name__ == "__main__":
    main()


