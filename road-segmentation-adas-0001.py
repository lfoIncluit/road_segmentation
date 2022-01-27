import cv2
import numpy as np
from openvino.inference_engine import IECore
from imutils import paths

TEST_PATH = "Images"
PAINT = True

pColor = (0, 0, 255)
rectThinkness = 1
alpha = 0.9

road_segmentation_model_xml = "./model/road-segmentation-adas-0001.xml"
road_segmentation_model_bin = "./model/road-segmentation-adas-0001.bin"

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


def road_segmentationDetection(
    frame,
    road_segmentation_neural_net,
    road_segmentation_execution_net,
    road_segmentation_input_blob,
    road_segmentation_output_blob,
    colormap,
):

    detections = []
    N, C, H, W = road_segmentation_neural_net.input_info[
        road_segmentation_input_blob
    ].tensor_desc.dims
    resized_frame = cv2.resize(frame, (W, H))
    image_h, image_w, _ = frame.shape

    # reshape to network input shape
    # Change data layout from HWC to CHW
    input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

    road_segmentation_results = road_segmentation_execution_net.infer(
        inputs={road_segmentation_input_blob: input_image}
    ).get(road_segmentation_output_blob)

    # Prepare data for visualization
    segmentation_mask = np.argmax(road_segmentation_results, axis=1)

    # search elements in the matrix
    road_segment_objects = ["background", "road", "curb", "lanemark"]
    elem_num = 0
    for elem in road_segment_objects:
        elem_num += 1
        if any(elem_num in sub for sub in segmentation_mask[0]):
            detections.append({"group": "road_segment", "type": elem})

    # Use function from notebook_utils.py to transform mask to an RGB image
    mask = segmentation_map_to_image(segmentation_mask, colormap)
    resized_mask = cv2.resize(mask, (image_w, image_h))
    cv2.imshow("mask", mask)
    cv2.waitKey(0)

    # Create image with mask put on
    image_with_mask = cv2.addWeighted(resized_mask, alpha, frame, 0.8, 0)

    cv2.imshow("image_with_mask", image_with_mask)


def main():

    ie = IECore()

    road_segmentation_neural_net = ie.read_network(
        model=road_segmentation_model_xml, weights=road_segmentation_model_bin
    )
    if road_segmentation_neural_net is not None:
        road_segmentation_input_blob = next(
            iter(road_segmentation_neural_net.input_info)
        )
        road_segmentation_output_blob = next(iter(road_segmentation_neural_net.outputs))
        road_segmentation_neural_net.batch_size = 1
        road_segmentation_execution_net = ie.load_network(
            network=road_segmentation_neural_net, device_name=device.upper()
        )

    # print(road_segmentation_neural_net.input_info[road_segmentation_input_blob].input_data.shape)
    # Define colormap BGR , each color represents a class:
    # background: black, road: purple, curb: orange, lanemark: yellow
    colormap = np.array([[0, 0, 0], [153, 76, 0], [0, 0, 255], [0, 255, 0]])

    # Define the transparency of the segmentation mask on the photo

    for imagePath in paths.list_images(TEST_PATH):
        print(imagePath)
        img = cv2.imread(imagePath)
        if img is None:
            continue

        road_segmentationDetection(
            img,
            road_segmentation_neural_net,
            road_segmentation_execution_net,
            road_segmentation_input_blob,
            road_segmentation_output_blob,
            colormap,
        )
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
