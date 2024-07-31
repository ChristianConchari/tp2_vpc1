import numpy as np

def frequency_domain_blur_measure(image: np.ndarray) -> float:
    """
    This function calculates the FM (Frequency Domain Image Blur Measure) of an image.
    
    Args:
    image (np.ndarray): The input image.
    
    Returns:
    float: The FM value of the input image.
    """
    # Step 1: Compute F which is the Fourier Transform representation of image I
    F = np.fft.fft2(image)
    
    # Step 2: Find Fc which is obtained by shifting the origin of F to center
    Fc = np.fft.fftshift(F)
    
    # Step 3: Calculate AF = abs(Fc) where AF is the absolute value of the centered Fourier transform of image I
    AF = np.abs(Fc)
    
    # Step 4: Calculate M = max(AF) where M is the maximum value of the frequency component in F
    M = np.max(AF)
    
    # Step 5: Calculate TH = the total number of pixels in F whose pixel value > thres, where thres = M/1000
    thres = M / 1000
    TH = np.sum(AF > thres)
    
    # Step 6: Calculate Image Quality measure (FM) from equation (1)
    img_height, img_width = image.shape
    FM = TH / (img_height * img_width)
    
    return FM
