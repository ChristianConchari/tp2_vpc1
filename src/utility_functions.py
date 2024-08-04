import matplotlib.pyplot as plt
from typing import List, Dict
from .video_processing_functions import detect_max_focus_points, display_video_highlight_max_focus

def plot_focus_measurement_versus_frame(
    quality_measurements_list: List,
    max_focus_frames: List, 
    max_focus_value: float
    ) -> None:
    """
    This function plots the quality measure curve and highlights the max focus frames and max focus value.
    
    Parameters:
    - quality_measurements_list (List): The list of quality measurements per frame.
    - max_focus_frames (List): The list of max focus frames.
    - max_focus_value (float): The max focus value.
    
    Returns:
    - None
    """
    # Find the max focus value and its index    
    max_focus_index = quality_measurements_list.index(max_focus_value)
    max_focus_value = round(max_focus_value, 4)
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Plot the quality measure curve
    plt.plot(quality_measurements_list, label='Quality per Frame', color='blue', zorder=1)

    # Highlight the max focus frames
    for i in max_focus_frames:
        plt.scatter(i, quality_measurements_list[i], color='green', zorder=3, edgecolors='black', s=20)
    
    # Add sample lines for the legend
    plt.scatter([], [], color='green', label='Max Focus Frames', edgecolors='black', s=20)
    
    # Highlight the max focus value
    plt.scatter(max_focus_index, max_focus_value, color='red', label='Max focus', zorder=5, edgecolors='black', s=20)
    plt.text(130, max_focus_value - (0.05*max_focus_value), f'  Frame {max_focus_index}\n  Max quality {max_focus_value}', verticalalignment='bottom', fontsize=12)

    # Add labels, title, and legend with increased font size
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Quality Measure', fontsize=12)
    plt.title('Image Quality Measure FM Curve\n', fontsize=14)

    # Add grid lines
    plt.grid(True)

    # Add legend
    plt.legend(fontsize=12)

    # Show the plot
    plt.show()
    
def plot_quality_measurements_on_2x2_grid(variations: List[Dict[str, int | float]], video_path: str) -> None:
    """
    Plots the quality measurements for different variations of NxM focus matrices
    on a 2x2 grid.
    
    Parameters:
    variations (list): A list of dictionaries with keys 'N', 'M', and 'roi_percentage'.
    video_path (str): The path to the video file.
    """
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))

    for i, var in enumerate(variations):
        # Compute the quality measurements of the video with the focus matrix
        quality_measurements_list, max_focus_points, max_focus_value = detect_max_focus_points(
            video_path=video_path, 
            N=var['N'], 
            M=var['M'],
            roi_percentage=var['roi_percentage']
        )
        
        # Find the max focus frames
        max_focus_frames = [value[0] for value in max_focus_points]
        
        # Find the max focus value and its index    
        max_focus_index = quality_measurements_list.index(max_focus_value)
        max_focus_value = round(max_focus_value, 4)

        # Select the subplot
        ax = axs[i//2, i%2]

        # Graph the quality measure curve
        ax.plot(quality_measurements_list, label='Quality per Frame', color='blue', zorder=1)
        
        # Highlight the max focus frames
        for frame in max_focus_frames:
            ax.scatter(frame, quality_measurements_list[frame], color='green', zorder=3, edgecolors='black', s=20)
            
        # Add sample lines for the legend
        ax.scatter([], [], color='green', label='Max Focus Frames', edgecolors='black', s=20)

        # Highlight the max focus value
        ax.scatter(max_focus_index, max_focus_value, color='red', label='Max focus', zorder=5, edgecolors='black', s=20)
        ax.text(125, max_focus_value - (0.05*max_focus_value), f'  Frame {max_focus_index}\n  Max quality {max_focus_value}', verticalalignment='bottom', fontsize=14)

        # Add labels, title, and legend
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Quality Measure', fontsize=12)
        ax.set_title(f'Image Quality Measure FM Curve (N={var["N"]}, M={var["M"]})', fontsize=14)

        # Add grid lines
        ax.grid(True)

        # Add legend
        ax.legend(fontsize=12)

    # Adjust the layout
    plt.tight_layout()
    plt.show()
    
def show_video_highlight_on_2x2_grid(variations: List[Dict[str, int | float]], video_path: str,) -> None:
    """
    Displays the video with the focus points highlighted on a 2x2 grid.
    
    Parameters:
    - variations (list): A list of dictionaries with keys 'N', 'M', and 'roi_percentage'.
    - video_path (str): The path to the
    
    Returns:
    - None
    """
    for i, var in enumerate(variations):
        # Compute the quality measurements of the video with the focus matrix
        quality_measurements_list, max_focus_points, max_focus_value = detect_max_focus_points(
            video_path=video_path, 
            N=var['N'], 
            M=var['M'],
            roi_percentage=var['roi_percentage']
        )
        
        # Display the video with the focus points highlighted
        display_video_highlight_max_focus(
            video_path = video_path, 
            M = var['N'],
            N = var['M'],
            roi_percentage = var['roi_percentage'],
            max_focus_points = max_focus_points, 
            quality_measurements_list = quality_measurements_list, 
            title = 'Frame with Focus Highlight\n' + f'N={var["N"]}, M={var["M"]}'   
        )