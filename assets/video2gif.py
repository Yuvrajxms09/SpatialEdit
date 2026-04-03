import imageio

def video_to_gif(input_path, output_path="output.gif", fps=10, speed=2):
    reader = imageio.get_reader(input_path)
    frames = []

    for i, frame in enumerate(reader):
        if i % 10 == 0:
            frames.append(frame)

    # 🚀 提高播放速度
    imageio.mimsave(output_path, frames, fps=fps * speed)

    print("Saved to", output_path)


# 使用
video_to_gif("assets/application/moving/video.mp4", "assets/application/moving/video.gif", fps=10, speed=2)