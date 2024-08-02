import cv2 as cv
import numpy as np
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

def calculate_video_quality_grid(video_path: str, delay: int = 42, N: int = 7, M: int = 7, roi_percentage: float = 0.1) -> list:
    """
    Calculates the quality measure for each rectangular element in an NxM grid over a specified ROI for each frame in the video
    and displays the video in grayscale with the grid overlayed.
    
    Parameters:
    video_path (str): The path to the video file.
    delay (int): The delay between frames in milliseconds.
    N (int): Number of vertical grid elements.
    M (int): Number of horizontal grid elements.
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
    
    frame_number = 0

    while True:
        # Capture the first frame
        ret, frame = cap.read()    
        frame_number += 1  
                            
        # Check if the frame is captured successfully
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Get the height and width of the frame
        height, width = gray.shape
        
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
                fm = frequency_domain_blur_measure(elem)
                
                # Append the focus measure to the list
                frame_focus_measures.append(fm)
                
                # Draw a rectangle around the current grid element
                cv.rectangle(frame, (elem_x_start, elem_y_start), (elem_x_end, elem_y_end), (0, 255, 0), 1)
        
        # Calculate the average focus measure for the frame
        avg_focus_measure = np.mean(frame_focus_measures)
        
        # Append the average focus measure to the frame_focus_measures list
        quality_measure_list.append(avg_focus_measure)
        
        # Write the frame number and average focus measure on the frame
        text = f"Frame: {frame_number}, FM: {avg_focus_measure:.4f}"
        cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv.LINE_AA)
        
        # Show the frame with the grid overlay
        cv.imshow('Frame with Grid', frame)
        
        # Break the loop when 'q' is pressed or wait for the delay
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    
    # Destroy all the windows
    cv.destroyAllWindows()
    
    return quality_measure_list