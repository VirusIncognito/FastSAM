import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import image_to_np_ndarray
from PIL import Image

try:
    import clip  # for linear_assignment

except (ImportError, AssertionError, AttributeError):
    from ultralytics.yolo.utils.checks import check_requirements

    check_requirements('git+https://github.com/openai/CLIP.git')  # required before installing lap from source
    import clip


class FastSAMPrompt:

    def __init__(self, image, results, device='cuda'):
        if isinstance(image, str) or isinstance(image, Image.Image):
            image = image_to_np_ndarray(image)
        self.device = device
        self.results = results
        self.img = image
    
    def _segment_image(self, image, bbox):
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        segmented_image_array = np.zeros_like(image_array)
        x1, y1, x2, y2 = bbox
        segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
        segmented_image = Image.fromarray(segmented_image_array)
        black_image = Image.new('RGB', image.size, (255, 255, 255))
        # transparency_mask = np.zeros_like((), dtype=np.uint8)
        transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        transparency_mask[y1:y2, x1:x2] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image

    def _format_results(self, result, filter=0):
        annotations = []
        n = len(result.masks.data)
        for i in range(n):
            annotation = {}
            mask = result.masks.data[i] == 1.0

            if torch.sum(mask) < filter:
                continue
            annotation['id'] = i
            annotation['segmentation'] = mask.cpu().numpy()
            annotation['bbox'] = result.boxes.data[i]
            annotation['score'] = result.boxes.conf[i]
            annotation['area'] = annotation['segmentation'].sum()
            annotations.append(annotation)
        return annotations

    def filter_masks(annotations):  # filte the overlap mask
        annotations.sort(key=lambda x: x['area'], reverse=True)
        to_remove = set()
        for i in range(0, len(annotations)):
            a = annotations[i]
            for j in range(i + 1, len(annotations)):
                b = annotations[j]
                if i != j and j not in to_remove:
                    # check if
                    if b['area'] < a['area']:
                        if (a['segmentation'] & b['segmentation']).sum() / b['segmentation'].sum() > 0.8:
                            to_remove.add(j)

        return [a for i, a in enumerate(annotations) if i not in to_remove], to_remove

    def _get_bbox_from_mask(self, mask):
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2, y2 = x1 + w, y1 + h
        if len(contours) > 1:
            for b in contours:
                x_t, y_t, w_t, h_t = cv2.boundingRect(b)
                # Merge multiple bounding boxes into one.
                x1 = min(x1, x_t)
                y1 = min(y1, y_t)
                x2 = max(x2, x_t + w_t)
                y2 = max(y2, y_t + h_t)
            h = y2 - y1
            w = x2 - x1
        return [x1, y1, x2, y2]

    def plot_to_result(self,
             annotations,
             bboxes=None,
             points=None,
             point_label=None,
             better_quality=True,
             retina=False,
             withContours=True) -> np.ndarray:
        if isinstance(annotations[0], dict):
            annotations = [annotation['segmentation'] for annotation in annotations]
        image = self.img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h = image.shape[0]
        original_w = image.shape[1]
        if sys.platform == "darwin":
            plt.switch_backend("TkAgg")
        plt.figure(figsize=(original_w / 100, original_h / 100))
        # Add subplot with no margin.
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.imshow(image)
        if better_quality:
            if isinstance(annotations[0], torch.Tensor):
                annotations = np.array(annotations.cpu())
            for i, mask in enumerate(annotations):
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
        if self.device == 'cpu':
            annotations = np.array(annotations)
            self.fast_show_mask(
                annotations,
                plt.gca(),
                bboxes=bboxes,
                points=points,
                pointlabel=point_label,
                retinamask=retina,
                target_height=original_h,
                target_width=original_w,
            )
        else:
            if isinstance(annotations[0], np.ndarray):
                annotations = torch.from_numpy(annotations)
            self.fast_show_mask_gpu(
                annotations,
                plt.gca(),
                bboxes=bboxes,
                points=points,
                pointlabel=point_label,
                retinamask=retina,
                target_height=original_h,
                target_width=original_w,
            )
        if isinstance(annotations, torch.Tensor):
            annotations = annotations.cpu().numpy()
        if withContours:
            contour_all = []
            temp = np.zeros((original_h, original_w, 1))
            for i, mask in enumerate(annotations):
                if type(mask) == dict:
                    mask = mask['segmentation']
                annotation = mask.astype(np.uint8)
                if not retina:
                    annotation = cv2.resize(
                        annotation,
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                contours, hierarchy = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_all.append(contour)
            cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
            color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
            contour_mask = temp / 255 * color.reshape(1, 1, -1)
            plt.imshow(contour_mask)

        plt.axis('off')
        fig = plt.gcf()
        plt.draw()

        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
        cols, rows = fig.canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 3)
        result = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        plt.close()
        return result
            
    # Remark for refactoring: IMO a function should do one thing only, storing the image and plotting should be seperated and do not necessarily need to be class functions but standalone utility functions that the user can chain in his scripts to have more fine-grained control. 
    def plot(self,
             annotations,
             output_path,
             bboxes=None,
             points=None,
             point_label=None,
             better_quality=True,
             retina=False,
             withContours=True):
        if len(annotations) == 0:
            return None
        result = self.plot_to_result(
            annotations, 
            bboxes, 
            points, 
            point_label, 
            better_quality, 
            retina, 
            withContours,
        )

        path = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(path):
            os.makedirs(path)
        result = result[:, :, ::-1]
        cv2.imwrite(output_path, result)

        if self.device == 'cpu':
            mask_image_path = self.fast_show_mask(
            annotations,
            plt.gca(),
            bboxes=bboxes,
            points=points,
            pointlabel=point_label,
            retinamask=retina,
            target_height=result.shape[0],
            target_width=result.shape[1]
            )
        else:
            mask_image_path = self.fast_show_mask_gpu(
            annotations,
            plt.gca(),
            bboxes=bboxes,
            points=points,
            pointlabel=point_label,
            retinamask=retina,
            target_height=result.shape[0],
            target_width=result.shape[1]
            )

        return mask_image_path
     
    #   CPU post process
    def fast_show_mask(
        self,
        annotation,
        ax,
        bboxes=None,
        points=None,
        pointlabel=None,
        retinamask=True,
        target_height=960,
        target_width=960,
    ):
        mask_sum = annotation.shape[0]
        height = annotation.shape[1]
        weight = annotation.shape[2]
        #Sort annotations based on area.
        areas = np.sum(annotation, axis=(1, 2))
        sorted_indices = np.argsort(areas)
        annotation = annotation[sorted_indices]

        index = (annotation != 0).argmax(axis=0)

        # Fixed color (e.g., red)
        fixed_color = np.array([255 / 255, 255 / 255, 255 / 255])
        color = np.ones((mask_sum, 1, 1, 3)) * fixed_color

        transparency = np.ones((mask_sum, 1, 1, 1)) * 1
        visual = np.concatenate([color, transparency], axis=-1)
        mask_image = np.expand_dims(annotation, -1) * visual
        annotation = np.where(annotation, 255, 0).astype(np.uint8)
        if annotation.shape[0] == 1:
            annotation = annotation[0]
        output_folder = "output"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file_path = os.path.join(output_folder, "mask_image_mask.png")
        cv2.imwrite(output_file_path, annotation)

        show = np.zeros((height, weight, 4))
        h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing='ij')
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
        # Use vectorized indexing to update the values of 'show'.
        show[h_indices, w_indices, :] = mask_image[indices]
        if bboxes is not None:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
        # draw point
        if points is not None:
            plt.scatter(
                [point[0] for i, point in enumerate(points) if pointlabel[i] == 1],
                [point[1] for i, point in enumerate(points) if pointlabel[i] == 1],
                s=20,
                c='y',
            )
            plt.scatter(
                [point[0] for i, point in enumerate(points) if pointlabel[i] == 0],
                [point[1] for i, point in enumerate(points) if pointlabel[i] == 0],
                s=20,
                c='m',
            )

        if not retinamask:
            show = cv2.resize(show, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        ax.imshow(show)
        return output_file_path


    def fast_show_mask_gpu(
        self,
        annotation,
        ax,
        bboxes=None,
        points=None,
        pointlabel=None,
        retinamask=True,
        target_height=960,
        target_width=960,
    ):
        msak_sum = annotation.shape[0]
        height = annotation.shape[1]
        weight = annotation.shape[2]
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=False)
        annotation = annotation[sorted_indices]
        # Find the index of the first non-zero value at each position.
        index = (annotation != 0).to(torch.long).argmax(dim=0)

        # Fixed color (e.g., red)
        fixed_color = torch.tensor([255 / 255, 0 / 255, 0 / 255]).to(annotation.device)
        color = torch.ones((msak_sum, 1, 1, 3)).to(annotation.device) * fixed_color

        transparency = torch.ones((msak_sum, 1, 1, 1)).to(annotation.device) * 1
        visual = torch.cat([color, transparency], dim=-1)
        mask_image = torch.unsqueeze(annotation, -1) * visual

        annotation = np.where(annotation, 255, 0).astype(np.uint8)
        if annotation.shape[0] == 1:
            annotation = annotation[0]
        output_folder = "output"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file_path = os.path.join(output_folder, "mask_image_mask.png")
        cv2.imwrite(output_file_path, annotation)

        # Select data according to the index. The index indicates which batch's data to choose at each position, converting the mask_image into a single batch form.
        show = torch.zeros((height, weight, 4)).to(annotation.device)
        try:
            h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight), indexing='ij')
        except:
            h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
        # Use vectorized indexing to update the values of 'show'.
        show[h_indices, w_indices, :] = mask_image[indices]
        show_cpu = show.cpu().numpy()
        if bboxes is not None:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
        # draw point
        if points is not None:
            plt.scatter(
                [point[0] for i, point in enumerate(points) if pointlabel[i] == 1],
                [point[1] for i, point in enumerate(points) if pointlabel[i] == 1],
                s=20,
                c='y',
            )
            plt.scatter(
                [point[0] for i, point in enumerate(points) if pointlabel[i] == 0],
                [point[1] for i, point in enumerate(points) if pointlabel[i] == 0],
                s=20,
                c='m',
            )
        if not retinamask:
            show_cpu = cv2.resize(show_cpu, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        ax.imshow(show_cpu)
        return output_file_path

    # clip
    @torch.no_grad()
    def retrieve(self, model, preprocess, elements, search_text: str, device) -> int:
        preprocessed_images = [preprocess(image).to(device) for image in elements]
        tokenized_text = clip.tokenize([search_text]).to(device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = model.encode_image(stacked_images)
        text_features = model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = 100.0 * image_features @ text_features.T
        return probs[:, 0].softmax(dim=0)

    def _crop_image(self, format_results):

        image = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        ori_w, ori_h = image.size
        annotations = format_results
        mask_h, mask_w = annotations[0]['segmentation'].shape
        if ori_w != mask_w or ori_h != mask_h:
            image = image.resize((mask_w, mask_h))
        cropped_boxes = []
        cropped_images = []
        not_crop = []
        filter_id = []
        # annotations, _ = filter_masks(annotations)
        # filter_id = list(_)
        for _, mask in enumerate(annotations):
            if np.sum(mask['segmentation']) <= 100:
                filter_id.append(_)
                continue
            bbox = self._get_bbox_from_mask(mask['segmentation'])  # mask 的 bbox
            cropped_boxes.append(self._segment_image(image, bbox))  
            # cropped_boxes.append(segment_image(image,mask["segmentation"]))
            cropped_images.append(bbox)  # Save the bounding box of the cropped image.

        return cropped_boxes, cropped_images, not_crop, filter_id, annotations

    def text_prompt(self, text):
        if self.results == None:
            return []
        format_results = self._format_results(self.results[0], 0)
        cropped_boxes, cropped_images, not_crop, filter_id, annotations = self._crop_image(format_results)
        clip_model, preprocess = clip.load('ViT-B/32', device=self.device)
        scores = self.retrieve(clip_model, preprocess, cropped_boxes, text, device=self.device)
        max_idx = scores.argsort()
        max_idx = max_idx[-1]
        max_idx += sum(np.array(filter_id) <= int(max_idx))
        return np.array([annotations[max_idx]['segmentation']])

    def everything_prompt(self):
        if self.results == None:
            return []
        return self.results[0].masks.data
        
