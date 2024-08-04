import cv2 as cv
import numpy as np
from .quality_measure_functions import frequency_domain_blur_measure
from typing import Callable, Tuple, List

def detect_max_focus_points(
    video_path: str,
    quality_measure_function: Callable[[np.ndarray], float] = frequency_domain_blur_measure,
    N: int = 1,
    M: int = 1,
    roi_percentage: float = 1.0,
    max_focus_threshold: float = 0.85
) -> Tuple[List[float], List[Tuple[int, float]], float]:
    """
    Detects and returns the measure of focus for each frame in the video, 
    the frames with significant focus with their focus measures and 
    the maximum focus value.
    
    Parameters:
    - video_path (str): The path to the video file.
    - quality_measure_function (Callable[[np.ndarray], float]): The function used to calculate the quality measure.
    - N (int): Number of vertical grid elements.
    - M (int): Number of horizontal grid elements.
    - roi_percentage (float): The percentage of the frame to be considered as the region of interest.
    - max_focus_threshold (float): The threshold for significant focus frames.
    
    Returns:
    - List[float]: A list of quality measures for each frame.
    - List[Tuple[int, float]]: A list of tuples containing the frame number and fm for significant focus frames.
    - float: The maximum focus value.
    """
    # Create a video capture object
    cap = cv.VideoCapture(video_path)
    
    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return [], []
    
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
        
        # Get the height and width of the frame
        height, width = gray.shape
        
        # Handle the case where N, M, and roi_percentage are not provided
        if (N == 1 and M == 1 and roi_percentage == 1.0):
            # Compute the fm for the entire frame
            gray_roi = gray
        else:
            # Calculate the area of the region of interest
            roi_area = height * width * roi_percentage
            
            # Calculate the dimension of the region of interest based on the aspect ratio of NxM
            roi_height = int((roi_area / (M / N)) ** 0.5)
            roi_width = int(roi_height * (M / N))
            
            # Ensure ROI dimensions are within frame boundaries
            if roi_width > width:
                roi_width = width
                roi_height = int(roi_width * (N / M))
            
            if roi_height > height:
                roi_height = height
                roi_width = int(roi_height * (M / N))
            
            # Calculate the starting points of the region of interest
            x_start = max(0, width // 2 - roi_width // 2)
            y_start = max(0, height // 2 - roi_height // 2)
            x_end = min(width, x_start + roi_width)
            y_end = min(height, y_start + roi_height)
        
            # Crop the ROI from the frame
            gray_roi = gray[y_start:y_end, x_start:x_end]
            
        if N == 1 and M == 1:
            # Single ROI case
            fm = quality_measure_function(gray_roi)
            avg_focus_measure = fm
        else:
            # Calculate the size of each square element within the ROI
            roi_height, roi_width = gray_roi.shape
            elem_size = min(roi_height // (2 * N + 1), roi_width // (2 * M + 1)) 
            
            # Initialize a list to store the focus measures of each grid element
            frame_focus_measures  = []

            # Loop over each grid element
            for i in range(N):
                for j in range(M):
                    # Calculate the coordinates of the current grid element
                    elem_y_start = y_start + (2 * i + 1) * elem_size
                    elem_y_end = elem_y_start + elem_size
                    elem_x_start = x_start + (2 * j + 1) * elem_size
                    elem_x_end = elem_x_start + elem_size
                    
                    # Crop the current grid element from the ROI
                    elem = gray[elem_y_start:elem_y_end, elem_x_start:elem_x_end]
                    
                    # Compute the focus measure for the current grid element
                    fm = quality_measure_function(elem)
                    
                    # Append the focus measure to the list
                    frame_focus_measures.append(fm)
            
            # Calculate the average focus measure for the frame
            avg_focus_measure = np.mean(frame_focus_measures)
            
        # Append the average focus measure to the frame_focus_measures list
        quality_measure_list.append(avg_focus_measure)
    
    # Release the capture
    cap.release()
    
    # Destroy all the windows
    cv.destroyAllWindows()
    
    # Find the max focus value
    max_focus_value = max(quality_measure_list)
    
    # Determine the threshold for significant focus frames
    threshold = max_focus_threshold * max_focus_value
    
    # Identify the frames with significant focus and their focus measures
    max_focus_points = [(i, fm) for i, fm in enumerate(quality_measure_list) if fm >= threshold]
    
    return quality_measure_list, max_focus_points, max_focus_value
    

def display_video_highlight_max_focus(
    video_path: str, 
    max_focus_points: List[Tuple[int, float]],
    quality_measurements_list: List[float],
    M: int = 1,
    N: int = 1,
    roi_percentage: float = 1.0,
    delay: int = 42
    ):
    # Create a video capture object
    cap = cv.VideoCapture(video_path)
    
    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    
    # Initialize the frame number
    frame_number = 0
    
    # Get max focus frames
    max_focus_frames = [i for i, _ in max_focus_points]

    while True:
        # Capture the first frame
        ret, frame = cap.read()    
                            
        # Check if the frame is captured successfully
        if not ret:
            break
        
        # Convert the frame to grayscale
        frame_number += 1
        
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Get the height and width of the frame
        height, width = gray.shape
        
        # Handle the case where N, M, and roi_percentage are not provided
        if (N == 1 and M == 1 and roi_percentage == 1.0):
            # Compute the fm for the entire frame
            gray_roi = gray
            # Define the starting points for the entire frame
            x_start, y_start, x_end, y_end = 0, 0, width, height
            # Determine the color based on whether the frame is a max focus frame
            color = (0, 255, 0) if frame_number in max_focus_frames else (0, 0, 255)
            # Draw a rectangle around the ROI
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
        else:
            # Calculate the area of the region of interest
            roi_area = height * width * roi_percentage
            
            # Calculate the dimension of the region of interest based on the aspect ratio of NxM
            roi_height = int((roi_area / (M / N)) ** 0.5)
            roi_width = int(roi_height * (M / N))
            
            # Ensure ROI dimensions are within frame boundaries
            if roi_width > width:
                roi_width = width
                roi_height = int(roi_width * (N / M))
            
            if roi_height > height:
                roi_height = height
                roi_width = int(roi_height * (M / N))
            
            # Calculate the starting points of the region of interest
            x_start = max(0, width // 2 - roi_width // 2)
            y_start = max(0, height // 2 - roi_height // 2)
            x_end = min(width, x_start + roi_width)
            y_end = min(height, y_start + roi_height)
        
            # Crop the ROI from the frame
            gray_roi = gray[y_start:y_end, x_start:x_end]
        
        if N == 1 and M == 1:
            # Single ROI case
            fm = quality_measurements_list[frame_number - 1]
            # Determine the color based on whether the frame is a max focus frame
            color = (0, 255, 0) if frame_number in max_focus_frames else (0, 0, 255)
            # Draw a rectangle around the ROI
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
        else:
            # Calculate the size of each square element within the ROI
            roi_height, roi_width = gray_roi.shape
            elem_size = min(roi_height // (2 * N + 1), roi_width // (2 * M + 1)) 
            
            # Initialize a list to store the focus measures of each grid element
            frame_focus_measures  = []
            max_focus_value = -1
            max_focus_coords = (0, 0, 0, 0)

            # Loop over each grid element
            for i in range(N):
                for j in range(M):
                    # Calculate the coordinates of the current grid element
                    elem_y_start = y_start + (2 * i + 1) * elem_size
                    elem_y_end = elem_y_start + elem_size
                    elem_x_start = x_start + (2 * j + 1) * elem_size
                    elem_x_end = elem_x_start + elem_size
                    
                    # Determine the color based on whether the frame is a max focus frame
                    color = (0, 255, 0) if frame_number in max_focus_frames else (0, 0, 255)
                    
                    # Draw a rectangle around the current grid element
                    cv.rectangle(frame, (elem_x_start, elem_y_start), (elem_x_end, elem_y_end), color, 1)
                    
                    # Get the fm value
                    fm = quality_measurements_list[frame_number - 1]
        
        # Write the frame number and focus status on the frame
        text = f"Frame: {frame_number}, FM: {fm:.4f}, Focus: {'Yes' if frame_number in max_focus_frames else 'No'}"
        cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv.LINE_AA)
        
        # Show the frame
        cv.imshow('Frame with Focus Highlight', frame)
        
        # Break the loop when 'q' is pressed or wait for the delay
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    
    # Destroy all the windows
    cv.destroyAllWindows()