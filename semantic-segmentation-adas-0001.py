import cv2
import numpy as np
from openvino.inference_engine import IECore
from imutils import paths

TEST_PATH = "Images"
PAINT = True

pColor = (0, 0, 255)
rectThinkness = 1

semantic_segmentation_model_xml = "./model/semantic-segmentation-adas-0001.xml"
semantic_segmentation_model_bin = "./model/semantic-segmentation-adas-0001.bin"

device = "CPU"


def segmentation_map_to_image(
    result: np.ndarray, colormap: np.ndarray, remove_holes=False
) -> np.ndarray:
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    :param result: A single network result after converting to pixel values in H,W or 1,H,W shape.
    :param colormap: A numpy array of shape (num_classes, 3) with an RGB value per class.
    :param remove_holes: If True, remove holes in the segmentation result.
    :return: An RGB image where each pixel is an int8 value according to colormap.
    """
    if len(result.shape) != 2 and result.shape[0] != 1:
        raise ValueError(
            f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
        )

    if len(np.unique(result)) > colormap.shape[0]:
        raise ValueError(
            f"Expected max {colormap[0]} classes in result, got {len(np.unique(result))} "
            "different output values. Please make sure to convert the network output to "
            "pixel values before calling this function."
        )
    elif result.shape[0] == 1:
        result = result.squeeze(0)

    result = result.astype(np.uint8)

    contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
    mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for label_index, color in enumerate(colormap):
        label_index_map = result == label_index
        label_index_map = label_index_map.astype(np.uint8) * 255
        contours, hierarchies = cv2.findContours(
            label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            mask,
            contours,
            contourIdx=-1,
            color=color.tolist(),
            thickness=cv2.FILLED,
        )

    return mask


def semantic_segmentationDetection(
    frame,
    semantic_segmentation_neural_net,
    semantic_segmentation_execution_net,
    semantic_segmentation_input_blob,
    semantic_segmentation_output_blob,
    colormap,
    alpha,
):

    detections = []
    N, C, H, W = semantic_segmentation_neural_net.input_info[
        semantic_segmentation_input_blob
    ].tensor_desc.dims
    resized_frame = cv2.resize(frame, (W, H))
    image_h, image_w, _ = frame.shape
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # reshape to network input shape
    # Change data layout from HWC to CHW
    input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

    semantic_segmentation_results = semantic_segmentation_execution_net.infer(
        inputs={semantic_segmentation_input_blob: input_image}
    ).get(semantic_segmentation_output_blob)

    # Prepare data for visualization
    segmentation_mask = np.argmax(semantic_segmentation_results, axis=1)

    # search elements in the matrix
    semantic_segment_objects = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic_light",
        "traffic_sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "ego_vehicle",
    ]
    elem_num = 0
    for elem in semantic_segment_objects:
        elem_num += 1
        if any(elem_num in sub for sub in segmentation_mask[0]):
            detections.append(elem)

    # Use function from notebook_utils.py to transform mask to an RGB image
    mask = segmentation_map_to_image(segmentation_mask, colormap)
    resized_mask = cv2.resize(mask, (image_w, image_h))
    cv2.imshow("mask", resized_mask)
    cv2.waitKey(0)

    # Create image with mask put on
    image_with_mask = cv2.addWeighted(resized_mask, alpha, rgb_image, 1 - alpha, 0)

    cv2.imshow("image_with_mask", image_with_mask)


def main():

    ie = IECore()

    semantic_segmentation_neural_net = ie.read_network(
        model=semantic_segmentation_model_xml, weights=semantic_segmentation_model_bin
    )
    if semantic_segmentation_neural_net is not None:
        semantic_segmentation_input_blob = next(
            iter(semantic_segmentation_neural_net.input_info)
        )
        semantic_segmentation_output_blob = next(
            iter(semantic_segmentation_neural_net.outputs)
        )
        semantic_segmentation_neural_net.batch_size = 1
        semantic_segmentation_execution_net = ie.load_network(
            network=semantic_segmentation_neural_net, device_name=device.upper()
        )

    # print(semantic_segmentation_neural_net.input_info[semantic_segmentation_input_blob].input_data.shape)
    # Define colormap, each color represents a class:
    # background: white, semantic: blue, curb: green, lanemark: red
    colormap = np.array(
        [
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [145, 163, 176],  # blue-alice
            [245, 245, 220],  # beige
            [62, 174, 177],  # blue-deck
            [206, 70, 118],  # bougainvillea
            [151, 127, 115],  # brown-buckskin
            [201, 174, 93],  # brown-buf1
            [201, 174, 171],  # brown-buf2
            [103, 49, 71],  # purple-burgundy1
            [84, 61, 63],  # purple-burgundy2
            [80, 64, 77],  # purple-burgundy3
            [145, 163, 176],  # gray-cadete1
            [129, 135, 139],  # gray-cadete2
            [138, 154, 91],  # green-spring1
            [147, 197, 146],  # green-spring2
            [126, 159, 46],  # green-spring3
            [96, 78, 151],  # purple3
        ]
    )

    # Define the transparency of the segmentation mask on the photo
    alpha = 0.2

    for imagePath in paths.list_images(TEST_PATH):
        print(imagePath)
        img = cv2.imread(imagePath)
        if img is None:
            continue

        semantic_segmentationDetection(
            img,
            semantic_segmentation_neural_net,
            semantic_segmentation_execution_net,
            semantic_segmentation_input_blob,
            semantic_segmentation_output_blob,
            colormap,
            alpha,
        )
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
