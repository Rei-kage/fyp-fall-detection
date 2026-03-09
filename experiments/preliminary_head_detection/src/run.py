from frame_extractor import extract_frames
from head_detection import detect_head
import cv2


VIDEO_PATH = "experiments/preliminary_head_detection/raw_videos/IMG_1704.MOV"

def main():

    

    for frame_index, frame in extract_frames(VIDEO_PATH):

        result = detect_head(frame)

        if result is not None:

            x, y, w, h, head_y = result

            print("Frame:", frame_index, "Head Y:", head_y)

            cv2.rectangle(
                frame,
                (x,y),
                (x + w, y + h),
                (0,255,0),
                2
            )

            cv2.circle(
                frame,
                (x + w//2, head_y),
                5,
                (255,0,0),
                -1
            )



            # if previous_head_y is not None:
            #     if abs(head_y - previous_head_y) > 150:
            #         head_y = None
            # previous_head_y = head_y
            # print("Frame", frame_index, "Head Y:", head_y)

            

        cv2.imshow("Head Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

      


if __name__ == "__main__":
    main()