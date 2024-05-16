import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def find_dominant_color(image, k=3):
    """使用 k-means 算法找到图像的主色"""
    data = np.array(image)
    pixels = data.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    centers = kmeans.cluster_centers_
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = centers[labels[np.argmax(counts)]]
    return dominant_color

def create_image_with_opacity_gradient(dominant_color, original_size, scale=0.4):
    """创建一个新图像并应用从中心到边缘的不透明度梯度变化"""
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    new_image = Image.new('RGBA', new_size, (0, 0, 0, 0))

    center_x, center_y = new_size[0] // 2, new_size[1] // 2
    max_distance = np.hypot(center_x, center_y)
    center_opacity = 255
    edge_opacity = int(center_opacity * 0.95)

    for x in range(new_size[0]):
        for y in range(new_size[1]):
            distance = np.hypot(x - center_x, y - center_y)
            opacity = int(edge_opacity + (center_opacity - edge_opacity) * (1 - distance / max_distance))
            new_image.putpixel((x, y), (*map(int, dominant_color), opacity))

    return new_image

if __name__ == "__main__":
    image_path = './pic.jpg'  
    image = Image.open(image_path)
    image = image.convert('RGB')  

    dominant_color = find_dominant_color(image)
    scaled_image = create_image_with_opacity_gradient(dominant_color, image.size)

    scaled_image.show()  
    scaled_image.save('Block1.png') 
