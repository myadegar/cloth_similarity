import os
import mimetypes
from PIL import Image
import numpy as np
import argparse
import torch
import warnings
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from cloth_segmentation.infer import load_segment_model, cloth_segment
# from rembg import remove, new_session

from deep_person_reid.torchreid.utils import FeatureExtractor
from deep_person_reid.torchreid import metrics

from diffusers import AutoencoderKL
import clip


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-100>"

class ClothSimilarity():
    def __init__(self, device, vae_model_path, similarity_method):
        self.device = torch.device(device)

        # segmentation model
        checkpoint_path = r'./cloth_segmentation/checkpoint/cloth_segm_u2net_latest.pth'
        self.segment_model = load_segment_model(checkpoint_path, self.device)
        # self.session = new_session(model_name="u2net_cloth_seg")

        # feature extractor model for similarity calculation
        if similarity_method.lower() == 'vae':  # vae model
            seed = 538282
            self.g_cuda = torch.Generator(device=self.device)
            self.g_cuda.manual_seed(seed)
            self.vae_feature_extractor = AutoencoderKL.from_pretrained(vae_model_path, revision="float16")
            self.vae_feature_extractor = self.vae_feature_extractor.to(self.device).half()
        elif similarity_method.lower() == 'reid':  # reid model
            self.reid_feature_extractor = FeatureExtractor(model_name='osnet_ain_x0_5',
                                                           model_path='./osnet_ain_x0_5_imagenet.pth', device=device)
        elif similarity_method.lower() == 'clip':  # clip method
            self.clip_model, self.clip_preprocess = clip.load('ViT-L/14', device=device)
        else:
            raise Exception('Unknown similarity method !')

    def cloth_segmentation(self, input_dir_path, output_dir_path, max_img_size=1024):
        result_data = {}
        os.makedirs(output_dir_path, exist_ok=True)
        files = os.listdir(input_dir_path)
        images_file = [file for file in files if str(mimetypes.guess_type(file)[0]).startswith('image')]
        if not images_file:
            raise Exception('There is no image file in directory !')
        images_path = [os.path.join(input_dir_path, image_file) for image_file in images_file]
        for image_path, image_file in zip(images_path, images_file):
            print(f'Process on file: {image_path}')
            result_data[image_file] = {}
            file_name, ext = os.path.splitext(image_file)
            input_image = Image.open(image_path).convert('RGB')
            torch.cuda.empty_cache()
            segment_images, mask_images = cloth_segment(input_image=input_image, net=self.segment_model,
                                                        device=self.device,
                                                        alpha_matting=False, post_process_mask=True, concat_parts=False,
                                                        max_img_size=max_img_size)
            ##
            segment_img = {}
            mask_img = {}
            crop_img = {}
            bbox_points = {}
            for idx, segment_image in enumerate(segment_images):
                cloth_mask_img = mask_images[idx]
                binary_image = cloth_mask_img.convert('1')
                sum_pixel = np.sum(binary_image)
                H, W = segment_image.size
                pixel_ratio = sum_pixel / (H * W + 1)
                if pixel_ratio < 0.01:
                    continue
                cloth_bbox = segment_image.getbbox()
                cloth_segment_img = segment_image.crop(cloth_bbox)
                cloth_crop_img = input_image.crop(cloth_bbox)
                ##
                if idx == 0:  # upper body
                    segment_img['upper'] = cloth_segment_img
                    mask_img['upper'] = cloth_mask_img
                    crop_img['upper'] = cloth_crop_img
                    bbox_points['upper'] = cloth_bbox
                elif idx == 1:  # lower body
                    segment_img['lower'] = cloth_segment_img
                    mask_img['lower'] = cloth_mask_img
                    crop_img['lower'] = cloth_crop_img
                    bbox_points['lower'] = cloth_bbox
                elif idx == 2:  # other part
                    segment_img['other'] = cloth_segment_img
                    mask_img['other'] = cloth_mask_img
                    crop_img['other'] = cloth_crop_img
                    bbox_points['other'] = cloth_bbox
                    # modify upper segment
                    if 'upper' in segment_img.keys():
                        upper_segment_img = Image.composite(segment_images[2], segment_images[0], mask_images[2])
                        upper_cloth_mask_img = Image.composite(mask_images[2], mask_images[0], mask_images[2])
                        upper_cloth_bbox = upper_segment_img.getbbox()
                        upper_cloth_segment_img = upper_segment_img.crop(upper_cloth_bbox)
                        upper_cloth_crop_img = input_image.crop(upper_cloth_bbox)
                        ##
                        segment_img['upper'] = upper_cloth_segment_img
                        mask_img['upper'] = upper_cloth_mask_img
                        crop_img['upper'] = upper_cloth_crop_img
                        bbox_points['upper'] = upper_cloth_bbox
                    else:  # full body as upper
                        segment_img['upper'] = cloth_segment_img
                        mask_img['upper'] = cloth_mask_img
                        crop_img['upper'] = cloth_crop_img
                        bbox_points['upper'] = cloth_bbox

            ##
            result_data[image_file]['segment_img'] = segment_img
            result_data[image_file]['mask_img'] = mask_img
            result_data[image_file]['crop_img'] = crop_img
            result_data[image_file]['bbox_points'] = bbox_points
            #
            for body_part, img in segment_img.items():
                dir_path = os.path.join(output_dir_path, 'segment')
                os.makedirs(dir_path, exist_ok=True)
                output_path = os.path.join(dir_path, file_name + '_segment' + '_' + body_part + '.png')
                img.save(output_path)
            for body_part, img in mask_img.items():
                dir_path = os.path.join(output_dir_path, 'mask')
                os.makedirs(dir_path, exist_ok=True)
                output_path = os.path.join(dir_path, file_name + '_mask' + '_' + body_part + '.png')
                img.save(output_path)
            for body_part, img in crop_img.items():
                dir_path = os.path.join(output_dir_path, 'crop')
                os.makedirs(dir_path, exist_ok=True)
                output_path = os.path.join(dir_path, file_name + '_crop' + '_' + body_part + '.png')
                img.save(output_path)
        return result_data

    def encode_img_latents(self, images):
        torch.cuda.empty_cache()
        if not isinstance(images, list):
            images = [images]
        img_arr = np.stack([np.array(img) for img in images], axis=0)
        img_arr = img_arr / 255.0
        img_arr = torch.from_numpy(img_arr).half().permute(0, 3, 1, 2)
        img_arr = 2 * (img_arr - 0.5)
        latent_dists = self.vae_feature_extractor.encode(img_arr.to(self.device)).latent_dist
        latent_samples = latent_dists.sample(generator=self.g_cuda)
        latent_samples *= 0.18215
        return latent_samples

    def similarity_calculation(self, result_cloth, result_person, img_sim_mode, cloth_mode, similarity_method,
                               similarity_mode):
        img_sim_mode = img_sim_mode.lower() + '_img'
        cloth_image_files = []
        cloth_segment_images = []
        cloth_empty_image_files = []
        for image_file, data in result_cloth.items():
            segment_img = data[img_sim_mode].get(cloth_mode.lower())
            if segment_img is not None:
                cloth_image_files.append(image_file)
                segment_img = segment_img.convert(mode='RGB')
                cloth_segment_images.append(segment_img)
            else:
                cloth_empty_image_files.append(image_file)
                print(f'Warning: no segmentation result on {image_file}')
        ##
        person_image_files = []
        person_segment_images = []
        person_empty_image_files = []
        for image_file, data in result_person.items():
            segment_img = data[img_sim_mode].get(cloth_mode)
            if segment_img is not None:
                person_image_files.append(image_file)
                segment_img = segment_img.convert(mode='RGB')
                person_segment_images.append(segment_img)
            else:
                person_empty_image_files.append(image_file)
                print(f'Warning: no segmentation result on {image_file}')
        ##
        if not cloth_image_files or not person_image_files:
            all_person_image_files = person_image_files + person_empty_image_files
            return all_person_image_files, [-1] * len(all_person_image_files)
        ##
        if similarity_method.lower() == 'reid':
            cloth_array_segment_images = [np.array(cloth_img) for cloth_img in cloth_segment_images]
            person_array_segment_images = [np.array(person_img) for person_img in person_segment_images]
            features_cloth = self.reid_feature_extractor(cloth_array_segment_images)
            features_person = self.reid_feature_extractor(person_array_segment_images)
        elif similarity_method.lower() == 'vae':
            cloth_resize_segment_images = [cloth_img.resize((512, 512)) for cloth_img in cloth_segment_images]
            person_resize_segment_images = [person_img.resize((512, 512)) for person_img in person_segment_images]
            features_cloth = self.encode_img_latents(cloth_resize_segment_images)
            features_person = self.encode_img_latents(person_resize_segment_images)
            features_cloth = torch.reshape(features_cloth, (features_cloth.shape[0], -1)).detach()
            features_person = torch.reshape(features_person, (features_person.shape[0], -1)).detach()
        elif similarity_method.lower() == 'clip':
            with torch.no_grad():
                cloth_segment_images = [self.clip_preprocess(cloth_segment_image).unsqueeze(0).to(self.device) for
                                        cloth_segment_image in cloth_segment_images]
                cloth_segment_images = [
                    torch.tensor(self.clip_model.encode_image(cloth_segment_image), dtype=torch.float32) for
                    cloth_segment_image in cloth_segment_images]
                features_cloth = torch.concat(cloth_segment_images)
                person_segment_images = [self.clip_preprocess(person_segment_image).unsqueeze(0).to(self.device) for
                                         person_segment_image in person_segment_images]
                person_segment_images = [
                    torch.tensor(self.clip_model.encode_image(person_segment_image), dtype=torch.float32) for
                    person_segment_image in person_segment_images]
                features_person = torch.concat(person_segment_images)
        else:
            raise Exception('Incorrect similarity method !')
        ##
        similarity_matrix = metrics.distance.cosine_distance(features_cloth, features_person)
        similarity_matrix = 1.0 - similarity_matrix.cpu().numpy()
        ##
        if similarity_mode.lower() == 'mean':
            similarity_scores = np.mean(similarity_matrix, axis=0)
        elif similarity_mode.lower() == 'max':
            similarity_scores = np.max(similarity_matrix, axis=0)
        else:
            raise Exception('Incorrect similarity mode !')
        ##
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_similarity_scores = similarity_scores[sorted_indices]
        sorted_similarity_scores = [round(score, 3) for score in sorted_similarity_scores]
        sorted_person_image_files = [person_image_files[idx] for idx in sorted_indices]
        for image_file in person_empty_image_files:
            sorted_person_image_files.append(image_file)
            sorted_similarity_scores.append(-1)

        return sorted_person_image_files, sorted_similarity_scores

    def grid_maker(self, images_path, result_dir):
        col = 5
        row = math.ceil(len(images_path) / col)
        scale = 4
        fig, axes = plt.subplots(row, col, figsize=(col * scale, row * scale), gridspec_kw={'hspace': 0, 'wspace': 0})
        for j, image_path in enumerate(images_path):
            if row == 1:
                currAxes = axes[j]
            else:
                currAxes = axes[j // col, j % col]
            currAxes.set_title(f"Img {j}")
            if j == 0:
                currAxes.text(-0.1, 0.5, "samples", rotation=0, va='center', ha='center', transform=currAxes.transAxes)
            img = mpimg.imread(image_path)
            currAxes.imshow(img, cmap='gray')
            currAxes.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'sorted_grid.png'), dpi=150)

    def __call__(self, cloth_dir, person_dir, result_dir,
                 max_img_size=1024,
                 cloth_mode='upper',
                 img_sim_mode='segment',
                 similarity_method='reid',
                 similarity_mode='mean'
                 ):
        cloth_output_dir = os.path.join(result_dir, 'cloth')
        person_output_dir = os.path.join(result_dir, 'person')
        print('-' * 50)
        result_cloth = self.cloth_segmentation(input_dir_path=cloth_dir, output_dir_path=cloth_output_dir,
                                               max_img_size=max_img_size)
        print('-' * 50)
        result_person = self.cloth_segmentation(input_dir_path=person_dir, output_dir_path=person_output_dir,
                                                max_img_size=max_img_size)
        print('-' * 50)
        sorted_person_image_files, sorted_similarity_scores = self.similarity_calculation(result_cloth=result_cloth,
                                                                                          result_person=result_person,
                                                                                          img_sim_mode=img_sim_mode,
                                                                                          cloth_mode=cloth_mode,
                                                                                          similarity_method=similarity_method,
                                                                                          similarity_mode=similarity_mode)

        sorted_images_path = [os.path.join(person_dir, img) for img in sorted_person_image_files]
        self.grid_maker(sorted_images_path, result_dir)
        # print score in image
        # add batch size
        return sorted_person_image_files, sorted_similarity_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cloth_dir', type=str, default='./input_images/cloth',
                        help='Input directory path of cloth images.')
    parser.add_argument('-p', '--person_dir', type=str, default='./input_images/person',
                        help='Input directory path of person images.')
    parser.add_argument('-r', '--result_dir', type=str, default='./result_images', help='Output directory of result.')
    parser.add_argument('-m', '--max_size', type=int, default=1024,
                        help='resize image in segmentation if image size is more than max_size')
    parser.add_argument('-x', '--cloth_mode', type=str, default='upper',
                        help='segment part of body for similarity calculation (upper | lower)')
    parser.add_argument('-i', '--img_mode', type=str, default='segment',
                        help='kind of image for similarity check (segment | crop)')
    parser.add_argument('-y', '--sim_method', type=str, default='clip',
                        help='feature extraction method (reid | vae | clip)')
    parser.add_argument('-z', '--sim_mode', type=str, default='mean',
                        help='compare similarity mode between clothes and persons (mean | max)')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device for processing (cuda | cpu)')
    parser.add_argument('-v', '--vae_model', type=str, default='', help='Directory Path to vae model')
    args = parser.parse_args()
    ##
    cloth_dir = args.cloth_dir
    person_dir = args.person_dir
    result_dir = args.result_dir
    max_img_size = args.max_size
    cloth_mode = args.cloth_mode
    img_sim_mode = args.img_mode
    similarity_method = args.sim_method
    similarity_mode = args.sim_mode
    device = args.device
    vae_model_path = args.vae_model
    ##
    # max_img_size = 768
    # img_sim_mode = 'crop'

    print("\n ***** Selected args: *****")
    print(f'cloth_dir: {cloth_dir}')
    print(f'person_dir: {person_dir}')
    print(f'result_dir: {result_dir}')
    print(f'max_img_size: {max_img_size}')
    print(f'cloth_mode: {cloth_mode}')
    print(f'image_similarity_mode: {img_sim_mode}')
    print(f'similarity_method: {similarity_method}')
    print(f'similarity_mode: {similarity_mode}')
    print(f'device: {device}')
    print(f'vae_model_path: {vae_model_path}')
    print('-' * 50)
    ##
    cloth_similarity = ClothSimilarity(device=device, vae_model_path=vae_model_path,
                                       similarity_method=similarity_method)
    t0 = time()
    sorted_person_image_files, sorted_similarity_scores = cloth_similarity(cloth_dir=cloth_dir,
                                                                           person_dir=person_dir,
                                                                           result_dir=result_dir,
                                                                           max_img_size=max_img_size,
                                                                           cloth_mode=cloth_mode,
                                                                           img_sim_mode=img_sim_mode,
                                                                           similarity_method=similarity_method,
                                                                           similarity_mode=similarity_mode
                                                                           )
    t1 = time()
    ##
    print('-' * 50)
    print(f'Procesing time: {round(t1 - t0, 1)} sec')
    print('-' * 50)
    print('sorted image files: \n', sorted_person_image_files)
    print('sorted similarity scores: \n', sorted_similarity_scores)
