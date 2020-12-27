"""This module contains the particle filter for a tracking purpose."""

import cv2
import sys
import numpy as np
import argparse



OBJ_DIM = (50, 80)
CROSSHAIR_DIM = (80, 80)
PARTICLE_SIGMA = np.min([OBJ_DIM]) // 4
ESC = 27
RED = (0, 0, 255)
GREEN = (0, 255, 0)


def make_crosshairs(img, top_left, bottom_right, color_channel, captured):
    obj_height, obj_width = CROSSHAIR_DIM
    center_x, center_y = img.shape[0] //2, img.shape[1] // 2

    img = cv2.rectangle(img, top_left, bottom_right, color_channel, 1)
    img = cv2.line(
            img,
            (center_x, img.shape[0] // 3),
            (center_x, center_y - obj_height // 2),
            color_channel, 1)
    img = cv2.line(
            img,
            (center_x, center_y + obj_height // 2),
            (center_x, img.shape[0] * 2 // 3),
            color_channel, 1)
    img = cv2.line(
            img,
            (img.shape[1] // 3, center_y),
            (center_x - obj_width // 2, center_y),
            color_channel, 1)
    img = cv2.line(
            img,
            (center_x + obj_width // 2, center_y),
            (img.shape[1] * 2 // 3, center_y),
            color_channel, 1)
    return img


def mark_target(img, center_xy, color_channel, captured):
    """Mark the target."""
    obj_height, obj_width = OBJ_DIM
    center_x, center_y = int(center_xy[0]), int(center_xy[1])

    top_left_x = int(center_xy[0] - OBJ_DIM[1] // 2)
    top_left_y = int(center_xy[1] - OBJ_DIM[0] // 2)
    bottom_right_x = int(center_xy[0] + OBJ_DIM[1] // 2)
    bottom_right_y = int(center_xy[1] + OBJ_DIM[0] // 2)

    img = cv2.rectangle(img,
            (top_left_x, top_left_y),
            (bottom_right_x, bottom_right_y),
            color_channel,
            1
        )

    img = cv2.line(
            img,
            (center_x, 0), (center_x, center_y - obj_height // 2),
            color_channel, 1)
    img = cv2.line(
            img,
            (center_x, center_y + obj_height // 2), (center_x, img.shape[0]),
            color_channel, 1)
    img = cv2.line(
            img,
            (0, center_y), (center_x - obj_width // 2, center_y),
            color_channel, 1)
    img = cv2.line(
            img,
            (center_x + obj_width // 2, center_y), (img.shape[1], center_y),
            color_channel, 1)
    return img


def reset_particles_vars(particles_xy, particles_scores, particles_patches):
    return [], [], []


def introduce_noise(particles_xy, img_height, img_width):
    """Introduce noise for the next generation."""
    for i, p in enumerate(particles_xy):
        if i == 0:
            continue
        p[0] += np.random.normal(0, PARTICLE_SIGMA)
        p[1] += np.random.normal(0, PARTICLE_SIGMA)

        # adjust for out_of_frame particles
        p[0] = OBJ_DIM[1] // 2 if p[0] < OBJ_DIM[1] else p[0]
        p[0] = img_width - OBJ_DIM[1] // 2 if p[0] > img_width - OBJ_DIM[1] // 2 else p[0]
        p[1] = OBJ_DIM[0] // 2 if p[1] < OBJ_DIM[0] // 2 else p[1]
        p[1] = img_height - OBJ_DIM[0] // 2 if p[1] > img_width - OBJ_DIM[0] // 2 else p[1]
    return particles_xy


def display_to_check(particles_xy, img_color):
    """Display the particles to check."""
    for p in particles_xy:
        img_color = cv2.circle(img_color,
                (int(p[0]), int(p[1])), 1, GREEN, -1)
    return img_color


def get_patches(particles_xy, img_color_clean):
    """Get patches for each particles."""
    particles_patches = []
    for p in particles_xy:
        patch_top_left_x = int(p[0] - OBJ_DIM[1] // 2)
        patch_top_left_y = int(p[1] - OBJ_DIM[0] // 2)
        patch_bottom_right_x = int(p[0] + OBJ_DIM[1] // 2)
        patch_bottom_right_y = int(p[1] + OBJ_DIM[0] // 2)
        temp_patch = img_color_clean[patch_top_left_y : patch_bottom_right_y,
                                     patch_top_left_x : patch_bottom_right_x]
        particles_patches.append(temp_patch)
    return particles_patches




def convert_to_proba(particles_scores):
    """Convert the scores to a probability."""
    particles_scores = np.array(particles_scores)
    particles_scores = 1.0 / (2.0 * np.pi * ARGS.distribution_sigma) * np.exp(-particles_scores / (2.0 * ARGS.distribution_sigma ** 2))
    return particles_scores / np.sum(particles_scores)


def compare_patches(img_patch, particles_patches, particles_scores):
    """Compare each patch with the model patch."""
    model_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
    model_patch = cv2.GaussianBlur(model_patch, (3, 3), 0)
    particles_scores = [] # a retirer
    for p in particles_patches:
        temp_patch = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        temp_patch = cv2.GaussianBlur(temp_patch, (3, 3), 0)
        mse = np.mean((model_patch - temp_patch) **2)
        particles_scores.append(mse)
    return convert_to_proba(particles_scores)


def resample(particles_xy, particles_scores):
    """Resampling function."""
    new_particles_xy_indexes = np.random.choice(
            range(ARGS.num_particles),
            size=ARGS.num_particles-1,
            p = particles_scores,
            replace=True
        )
    best_index = np.where(particles_scores == np.max(particles_scores))[0][0]
    best_xy = particles_xy[best_index]
    new_set = particles_xy[new_particles_xy_indexes]
    return np.vstack((best_xy, new_set)), best_xy, best_index


PARSER = argparse.ArgumentParser()

PARSER.add_argument(
    "--video_path",
    default="",
    help="Path to the video"
)

PARSER.add_argument(
    "--num_particles",
    default=200,
    help="Number of particle for the filter"
)

PARSER.add_argument(
    "--new_size_factor",
    default=0.4,
    help="The new size factor"
)

PARSER.add_argument(
    "--distribution_sigma",
    default = 0.5,
    help="The distribution sigma"
)


if __name__=="__main__":
    ARGS = PARSER.parse_args()
    cam = cv2.VideoCapture(r'{}'.format(ARGS.video_path))
    captured = False
    img_patch = np.zeros(OBJ_DIM)
    particles_xy, particles_scores, particles_patches = [], [], []
    ret, img = cam.read()
    if img is None:
        cam.release()
        sys.exit(0)

    img_color = cv2.resize(
        img,
        (int(img.shape[1]*ARGS.new_size_factor),
         int(img.shape[0]*ARGS.new_size_factor)))
    img_height, img_width, _ = img_color.shape
    top_left_x = img_width // 2 - OBJ_DIM[1] // 2
    top_left_y = img_height // 2 - OBJ_DIM[0] // 2
    bottom_right_x = img_width // 2 + OBJ_DIM[1] // 2
    bottom_right_y = img_height // 2 + OBJ_DIM[0] // 2

    while True:
        _, img = cam.read()
        if img is None:
            cam.release()
            sys.exit(0)
        img_color = cv2.resize(
           img,
           (int(img.shape[1]*ARGS.new_size_factor),
            int(img.shape[0]*ARGS.new_size_factor)))
        img_color_clean = img_color.copy()
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        if captured :
            key = cv2.waitKey(10) & 0xFF
        else:
            key = cv2.waitKey(25) & 0xFF
        if key == ESC :
            break

        if (key == ord("c")) or (key == ord("C")):
            captured = True
            img_patch = img_color_clean[
                    top_left_y : bottom_right_y,
                    top_left_x : bottom_right_x]
            particles_xy = np.zeros((ARGS.num_particles, 2))
            particles_xy[:, :] = [img_width // 2, img_height // 2]

        elif (key == ord("d")) or (key == ord("D")) :
            captured = False
            img_patch = np.zeros(img_patch.shape)

        if captured:
            particles_xy = introduce_noise(particles_xy, img_height, img_width)
            img_color = display_to_check(particles_xy, img_color)
            particles_patches = get_patches(particles_xy, img_color_clean)
            particles_scores = compare_patches(img_patch, particles_patches,
                    particles_scores)
            particles_xy, best_xy, best_index = resample(particles_xy,
                    particles_scores)
            print(particles_xy)

            # display best_xy / mark target
            img_color = mark_target(img_color, best_xy, RED, 1)

            # update model path
            img_patch = particles_patches[best_index]

        img_color = make_crosshairs(img_color,
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                GREEN,
                1
            )
        cv2.imshow("Object Tracker", img_color)

    cam.release()
    cv2.destroyAllWindows()


