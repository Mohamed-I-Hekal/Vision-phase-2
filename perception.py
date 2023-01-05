import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def impose_range(xpix, ypix, range=80):
    dist = np.sqrt(xpix**2 + ypix**2)
    return xpix[dist < range], ypix[dist < range]

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    #debugger flag
    debugger = False # to enable debugging mode, set to True
    
    # 1) Define source and destination points for perspective transform
     # Define calibration box in source (actual) and destination (desired) coordinates
        # These source and destination points are defined to warp the image
        # to a grid where each 10x10 pixel square represents 1 square meter
        # The destination box will be 3*dst_size on each side
    dst_size = 3.5
        # Set a bottom offset to account for the fact that the bottom of the image
        # is not the position of the rover but a bit in front of it
        # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    image = Rover.img
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(image, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable_threshed = color_thresh(warped)
        # ignore half of the image as bad data
    # navigable[0:int(navigable.shape[0]/2), :] = 0

        # Obstacles are simply navigable inverted
    mask = np.ones_like(navigable_threshed)
    mask[:,:] = 255
    mask = perspect_transform(mask, source, destination)
    obs_map = np.absolute((np.float32(navigable_threshed)-1) * mask)
        # ignore half of the image as bad data
    # obstacles[0:int(obstacles.shape[0]/2),:] = 0
###################################################################### VV ##############################################
        # identify the rock
    lower_yellow = np.array([24 - 5, 100, 100])
    upper_yellow = np.array([24 + 5, 255, 255])
            # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Threshold the HSV image to get only upper_yellow colors
    rock_samples = cv2.inRange(hsv, lower_yellow, upper_yellow)
    rock_samples = perspect_transform(rock_samples, source, destination)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
       
###################################################################### ^^ ##############################################
    Rover.vision_image[:,:,0] = obs_map
    Rover.vision_image[:,:,1] = rock_samples
    Rover.vision_image[:,:,2] = navigable_threshed
    idx = np.nonzero(Rover.vision_image)
    Rover.vision_image[idx] = 255

    # 5) Convert map image pixel values to rover-centric coords
    xpix_navigable, ypix_navigable = rover_coords(navigable_threshed)
    xpix_obs, ypix_obs = rover_coords(obs_map)
    xpix_rocks, ypix_rocks = rover_coords(rock_samples)

    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10.5
    xpix_navigable, ypix_navigable = impose_range(xpix_navigable, ypix_navigable)
    xpix_obs, ypix_obs = impose_range(xpix_obs, ypix_obs)
    navigable_x_world, navigable_y_world = pix_to_world(xpix_navigable, ypix_navigable,
                                                        Rover.pos[0], Rover.pos[1],
                                                        Rover.yaw, Rover.worldmap.shape[0], scale)
    obs_x_world, obs_y_world = pix_to_world(xpix_obs, ypix_obs,
                                                      Rover.pos[0], Rover.pos[1],
                                                      Rover.yaw, Rover.worldmap.shape[0], scale)
    rock_x_world, rock_y_world = pix_to_world(xpix_rocks, ypix_rocks,
                                              Rover.pos[0], Rover.pos[1],
                                              Rover.yaw, Rover.worldmap.shape[0], scale)
###################################################################### VV ##############################################
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        

        # Only update map if pitch an roll are near zero
    if (Rover.pitch < 1 or Rover.pitch > 359) and (Rover.roll < 1 or Rover.roll > 359):
        # increment = 10
        Rover.worldmap[obs_y_world, obs_x_world, 0] = 255
        Rover.worldmap[rock_y_world, rock_x_world,1] = 255
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] = 255
            # remove overlap mesurements
        nav_pix = Rover.worldmap[:, :, 2] > 0
        Rover.worldmap[nav_pix, 0] = 0
            # clip to avoid overflow
        Rover.worldmap = np.clip(Rover.worldmap, 0, 255)

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles

    dist, angles = to_polar_coords(xpix_navigable, ypix_navigable)
    Rover.nav_dists = dist
    Rover.nav_angles = angles
        # Same for rock samples
    dist, angles = to_polar_coords(xpix_rocks, ypix_rocks)
    Rover.rocks_dists = dist
    Rover.rocks_angles = angles
    
    # debugger saves realtime images of autonomous mode in folder pipeline_realtime
    if debugger == True:
        fig = plt.figure(figsize=(12,9))
        plt.subplot(221)
        plt.imshow(image)
        plt.subplot(222)
        plt.imshow(warped)
        plt.subplot(223)
        plt.imshow(threshed, cmap='gray')
        plt.subplot(224)
        plt.plot(xpix, ypix, '.')
        plt.ylim(-160, 160)
        plt.xlim(0, 160)
        arrow_length = 100
        x_arrow = arrow_length * np.cos(mean_dir)
        y_arrow = arrow_length * np.sin(mean_dir)
        plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
        
        idx = np.random.randint(0, 999999999)
        fig.savefig('../pipeline_realtime/Image' + str(idx) + '.jpg')
        plt.close(fig)
    return Rover
###################################################################### ^^ ##############################################
