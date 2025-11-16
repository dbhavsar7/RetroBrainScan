from src.gradcam import generate_cam

if __name__ == "__main__":
    generate_cam(
        img_path="data/raw/test/Moderate Impairment/14.jpg",
        output_path="outputs/heatmap_6.png"
    )
