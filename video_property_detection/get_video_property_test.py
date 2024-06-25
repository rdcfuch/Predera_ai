from videoprops import get_video_properties

props = get_video_properties('/Users/fcfu/Downloads/runway_6.mp4')

print(f'''
Codec: {props['codec_name']}
Resolution: {props['width']}×{props['height']}
Aspect ratio: {props['display_aspect_ratio']}
Frame rate: {props['avg_frame_rate']}
''')
