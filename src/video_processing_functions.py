import cv2 as cv
from .quality_measure_functions import frequency_domain_blur_measure

def calculate_video_quality(video_path: str, delay: int = 42, roi_percentage: float = None) -> list:
    """
    Calculates the quality measure for each frame in the video and displays the video in grayscale.
    
    Parameters:
    video_path (str): The path to the video file.
    delay (int): The delay between frames in milliseconds.
    roi_percentage (float): The percentage of the frame to be considered as the region of interest.
    
    Returns:
    list: A list of quality measures for each frame.
    """
    # Create a video capture object
    cap = cv.VideoCapture(video_path)
    
    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    
    # Initialize the quality measure list
    quality_measure_list = []

    while True:
        # Capture the first frame
        ret, frame = cap.read()      
                            
        # Check if the frame is captured successfully
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        if roi_percentage:
            # Get the height and width of the frame
            height, width = gray.shape
            
            # Calculate the area of the region of interest
            roi_area = height * width * roi_percentage
            
            # Calculate the region of interest
            roi_dim = int(roi_area ** 0.5)
            
            # Calculate the starting points of the region of interest
            x_start = max(0, width // 2 - roi_dim // 2)
            y_start = max(0, height // 2 - roi_dim // 2)
            x_end = min(width, x_start + roi_dim)
            y_end = min(height, y_start + roi_dim)
        
            # Crop the ROI from the frame
            gray_roi = gray[y_start:y_end, x_start:x_end]
            
            # Compute the fm for the ROI
            fm = frequency_domain_blur_measure(gray_roi)
            
            # Draw the ROI on the frame
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            
            # Show the frame with ROI
            cv.imshow(f'Frame with ROI ({int(roi_percentage*100)}% of frame)', frame)
        else:
            # Show the frame
            cv.imshow('Frame', frame)
            
            # Compute the fm for each frame
            fm = frequency_domain_blur_measure(gray)
        
        # Save the fm value to quality measure list 
        quality_measure_list.append(fm)
        
        # Break the loop when 'q' is pressed or wait for the delay
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    
    # Destroy all the windows
    cv.destroyAllWindows()
    
    return quality_measure_list