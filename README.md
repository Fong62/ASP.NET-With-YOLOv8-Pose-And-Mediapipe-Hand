# ASP.NET tích hợp YOLOv8 Pose And Mediapipe Hand

## Giới Thiệu
Chào mừng bạn đến với Pose Detect và Hand Detect cho trò chơi Endless Run Game! Đây là website được xây dựng để giúp người chơi có thể điều khiển nhân vật trò chơi Subway Surfers nhảy cúi và qua trái qua phải. Và hiện tại chỉ có thể chơi Subway Surfers.

## Tính Năng Chính
- **Phần camera tracking**: Tracking theo khung xương người dùng nếu là pose và tracking theo điểm giữa lòng bàn tay.
- **Di chuyển**: Pose thì di chuyển theo hướng di chuyển của cơ thể. Người chơi qua trái, phải thì nhân vật trong game cũng sẽ chạy qua trái, phải tương ứng với hướng di chuyển của người chơi và tương tự cho phần nhảy lên ngồi xuống.
- **Sử dụng ván trượt trong Subway Suffer**: Đối với HandDetect thì nắm bàn tay lại thì sẽ sử dụng ván trượt và cũng có thể start màn chơi khi nắm tay lại.
                                             Đối với PoseDetect thì dơ bàn tay lên thì sẽ sử dụng ván trượt và cũng có thể start màn chơi khi dơ tay lên.

## Công nghệ sử dụng
- **Trình biên dịch**: VSCode cho YOLOv8 Pose và Mediapipe Hand, VS Studio cho ASP.NET
- **Ngỗn ngữ**: Python, C#
- **Thư viện hỗ trợ**: Pygame, random, flask, opencv-python, cvzone, ultralytics, numpy, torch, mediapipe

## Yêu cầu hệ thống
- VS Studio, truy cập đường dẫn sau để tải về: https://code.visualstudio.com/.
- Tải python trên trang chủ pyhon về, truy cập đường dẫn sau để truy cập:https://www.python.org/downloads/.

## Khởi động Game
- Mở CMD tại thư mực Project chạy lệnh python app.py và mở thư mục App Tích hợp Yolov8 Pose_MediaPipe Hand trong VS Studio.
- Chạy ứng dụng để vào Web sau đó chọn YOLOv8 Pose hoặc Mediapipe Hand để sử dụng.

## Thông tin liên hệ
Gmail: nguyenhoangphongsupham@gmail.com

