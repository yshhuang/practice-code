from imagededup.methods import WHash
phasher = WHash()

# 生成图像目录中所有图像的二值hash编码
encodings = phasher.encode_images(image_dir='/Volumes/develop/data/lolita/kuaikan/image-326')

# 对已编码图像寻找重复图像
duplicates = phasher.find_duplicates(encoding_map=encodings)

# 给定一幅图像，显示与其重复的图像
from imagededup.utils import plot_duplicates

plot_duplicates(image_dir='/Volumes/develop/data/lolita/kuaikan/image-326',
                duplicate_map=duplicates,
                filename='686581155188703696-3')
