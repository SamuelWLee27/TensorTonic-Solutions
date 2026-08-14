def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride = image_size/feature_size
    boxes = []

    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            for scale in scales:
                for ratio in aspect_ratios:
                    w = scale * (ratio ** 0.5)
                    h = scale / (ratio ** 0.5)
                    boxes.append([cx - w/2, cy - h/2,  cx + w/2, cy + h/2])

    return boxes