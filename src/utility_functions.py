import matplotlib.pyplot as plt
from typing import List

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
    plt.text(max_focus_index + 32, max_focus_value - 0.08, f'  Frame {max_focus_index}\n  Max quality {max_focus_value}', verticalalignment='bottom', fontsize=14)

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