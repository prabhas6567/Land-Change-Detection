import sys
import os
import argparse
import torch
import logging
import time
from tqdm import tqdm
from image import *
from Models.lightUnetPlusPlus import lightUnetPlusPlus
from image import plot_accuracy_histogram
from torchvision import transforms


def predict(model,
            threshold,
            device,
            dataset,
            output_paths,
            color):

    with tqdm(desc=f'Prediction', unit=' img') as progress_bar:
        masks_predicted = []
        ground_truths = []
        for i, (image, ground_truth) in enumerate(dataset):
            image = image[0, ...]
            image = image.to(device)
            with torch.no_grad():
                mask_predicted = model(image)
            placeholder_path(output_paths[i])
            save_predicted_mask(mask_predicted, device, color=color, filename=(output_paths[i]+"/predicted.png"), threshold=threshold)
            masks_predicted.append(mask_predicted)
            ground_truths.append(ground_truth)
            progress_bar.update()

    # Unfold batched masks if necessary
    masks_predicted = unfold_batch(masks_predicted)
    ground_truths = unfold_batch(ground_truths)

    # Plot and save the histogram
    plot_accuracy_histogram(masks_predicted, ground_truths, filename="accuracy_histogram.png")


if __name__ == '__main__':
    t_start = time.time()
    current_path = sys.argv[0]
    current_path = current_path.replace("predict.py", "")


    # Hyperparameters
    batch_size = 1
    num_classes = 2
    n_channels = 6

    # Arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i")
    parser.add_argument("--output", "-o")                        
    parser.add_argument("--threshold", "-t", type=float)
    parser.add_argument("--color", "-c") # red blue black

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    logging.info(f'Using {device}')

    instance_names = [i for i in os.walk(args.input)][0][1]
    dataset, output_paths = load_dataset_predict(args.input, args.output, instance_names, batch_size)

    logging.info(f'Data loaded : {len(output_paths)} instances found')

    # Network creation, uncomment the one you want to use
    # model = BasicUnet(n_channels= n_channels, n_classes=num_classes)
    # model = modularUnet(n_channels=n_channels, n_classes=num_classes, depth=2)
    # model = unetPlusPlus(n_channels=n_channels, n_classes=num_classes)
    model = lightUnetPlusPlus(n_channels=n_channels, n_classes=num_classes)
    model.to(device)
    model.load_state_dict(torch.load('Weights/last.pth',map_location=torch.device(device)))
    model.eval()
    logging.info(f'Model loaded\n')

    transform = transforms.ToTensor()
    try:
        predict(model=model,
                threshold=args.threshold,
                device=device,
                dataset=dataset,
                output_paths=output_paths,
                color=args.color)


    except KeyboardInterrupt:
        logging.info(f'Interrupted by Keyboard')
    finally:
        t_end = time.time()
        print("\nDone in " + str(int((t_end - t_start))) + " sec")

# import sys
# import os
# import argparse
# import torch
# import logging
# import time
# from tqdm import tqdm
# from image import *
# from Models.lightUnetPlusPlus import lightUnetPlusPlus
# from image import plot_accuracy_histogram
# from torchvision import transforms
# from PIL import Image

# # Grad-CAM Imports
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image


# def predict(model,
#             threshold,
#             device,
#             dataset,
#             output_paths,
#             color):

#     # Choose target layer for Grad-CAM
#     target_layers = [model.decoder[-1]]  # Adjust this to your architecture

#     with tqdm(desc=f'Prediction', unit=' img') as progress_bar:
#         masks_predicted = []
#         ground_truths = []
#         for i, (image, ground_truth) in enumerate(dataset):
#             image = image[0, ...]
#             image = image.to(device)

#             with torch.no_grad():
#                 mask_predicted = model(image)

#             # Save predicted mask
#             placeholder_path(output_paths[i])
#             save_predicted_mask(mask_predicted, device, color=color, filename=(output_paths[i]+"/predicted.png"), threshold=threshold)
#             masks_predicted.append(mask_predicted)
#             ground_truths.append(ground_truth)

#             # ----- Grad-CAM Section -----
#             input_tensor = image.unsqueeze(0)  # Add batch dimension
#             cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

#             targets = [SemanticSegmentationTarget(1, ground_truth.squeeze().cpu().numpy())]
#             grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

#             # Convert image for display
#             rgb_img = image.cpu().numpy().transpose(1, 2, 0)
#             rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # Normalize to 0-1
#             cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

#             # Save Grad-CAM heatmap
#             cam_path = os.path.join(output_paths[i], "grad_cam.png")
#             Image.fromarray(cam_image).save(cam_path)

#             progress_bar.update()

#     masks_predicted = unfold_batch(masks_predicted)
#     ground_truths = unfold_batch(ground_truths)

#     plot_accuracy_histogram(masks_predicted, ground_truths, filename="accuracy_histogram.png")


# if __name__ == '__main__':
#     t_start = time.time()
#     current_path = sys.argv[0]
#     current_path = current_path.replace("predict.py", "")

#     batch_size = 1
#     num_classes = 2
#     n_channels = 6

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", "-i")
#     parser.add_argument("--output", "-o")                        
#     parser.add_argument("--threshold", "-t", type=float)
#     parser.add_argument("--color", "-c")
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Using {device}')

#     instance_names = [i for i in os.walk(args.input)][0][1]
#     dataset, output_paths = load_dataset_predict(args.input, args.output, instance_names, batch_size)
#     logging.info(f'Data loaded : {len(output_paths)} instances found')

#     model = lightUnetPlusPlus(n_channels=n_channels, n_classes=num_classes)
#     model.to(device)
#     model.load_state_dict(torch.load('Weights/last.pth', map_location=device))
#     model.eval()
#     print(model)
#     logging.info(f'Model loaded\n')

#     try:
#         predict(model=model,
#                 threshold=args.threshold,
#                 device=device,
#                 dataset=dataset,
#                 output_paths=output_paths,
#                 color=args.color)

#     except KeyboardInterrupt:
#         logging.info(f'Interrupted by Keyboard')
#     finally:
#         t_end = time.time()
#         print("\nDone in " + str(int((t_end - t_start))) + " sec")
