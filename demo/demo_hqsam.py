from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from pycocotools import mask
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm



def decode_rle(compressed_rle):
    decoded_rle = mask.decode(compressed_rle)
    return decoded_rle

def calculate_iou(mask1, mask2):   # From the question.
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and( mask1==1,  mask2==1 ))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou

if __name__ == '__main__':

    sam_checkpoint = "/home/vetoshkin_ln/text_sam_hq/checkpoints/sam_hq_vit_l.pth"
    model_type = "vit_l"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    
    predictor = SamPredictor(sam)

    annotation_file = 'annotations/val.json'
    images_filepath = '/home/vetoshkin_ln/text_sam_hq/train/data/val'
    coco = COCO(annotation_file)
    
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    m_iou = {}
    for name in cat_names:
        m_iou[name] = []
    
    img_ids = [info['id'] for info in coco.dataset['images']]

    for img_id in tqdm(img_ids):
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        image_name = coco.loadImgs(img_id)[0]['file_name'] #dataset['images'][img_id - 1]['file_name']
        image = cv2.imread(images_filepath + '/' + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for ann in anns:
            rle = ann['segmentation']
            category_id = ann['category_id']
            bbox = ann['bbox']
            compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
            binary_mask = decode_rle(compressed_rle)
            cat_name = coco.loadCats(category_id)[0]['name']

            input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=input_box[None, :],
                                multimask_output=False,
                            )
            category_id = ann['category_id']
            cat_name = coco.loadCats(category_id)[0]['name']

            iou = calculate_iou(masks[0], binary_mask)
            m_iou[cat_name].append(iou)
    for key in m_iou.keys():
        m_iou[key] = np.mean(m_iou[key])
    
    print (m_iou)
    