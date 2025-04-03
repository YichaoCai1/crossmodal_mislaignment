"""Generate text for the Multimodal3DIdent dataset."""

import argparse
import csv
import os

# define constants
XPOS = {
    0: 'left',
    1: 'center',
    2: 'right'}

YPOS = {
    0: 'top',
    1: 'mid',
    2: 'bottom'}

SHAPES = {
    0: 'teapot',
    1: 'hare',
    2: 'dragon',
    3: 'cow',
    4: 'armadillo',
    5: 'horse',
    6: 'head'}


PHRASES = \
[    
     {
        0: 'A {SHAPE} is visible.',
        1: 'A {SHAPE} is in the image.',
        2: 'The image shows a {SHAPE}.',
        3: 'The picture is a {SHAPE}.',
        4: 'There is an object in the form of a {SHAPE}.'
    },
    {
        0: 'A {SHAPE} is visible, positioned at the {XPOS} of the image.',
        1: 'A {SHAPE} is at the {XPOS} of the image.',
        2: 'The {XPOS} of the image shows a {SHAPE}.',
        3: 'At the {XPOS} of the picture is a {SHAPE}.',
        4: 'At the {XPOS} of the image, there is an object in the form of a {SHAPE}.'
    },
     {
        0: 'A {SHAPE} is visible, positioned at the {YPOS}-{XPOS} of the image.',
        1: 'A {SHAPE} is at the {YPOS}-{XPOS} of the image.',
        2: 'The {YPOS}-{XPOS} of the image shows a {SHAPE}.',
        3: 'At the {YPOS}-{XPOS} of the picture is a {SHAPE}.',
        4: 'At the {YPOS}-{XPOS} of the image, there is an object in the form of a {SHAPE}.'
    },
    {
        0: 'A {SHAPE} is visible, positioned at the {YPOS}-{XPOS} of the image, with a spotlight shining from the {SPOTLIGHT_POS}.',
        1: 'A {SHAPE} is at the {YPOS}-{XPOS} of the image, illuminated by a light from the {SPOTLIGHT_POS}.',
        2: 'The {YPOS}-{XPOS} of the image shows a {SHAPE}, highlighted by a light source positioned at the {SPOTLIGHT_POS}.',
        3: 'At the {YPOS}-{XPOS} of the picture is a {SHAPE}, standing out under a light coming from the {SPOTLIGHT_POS}.',
        4: 'At the {YPOS}-{XPOS} of the image, there is an object in the form of a {SHAPE}, with illumination from the {SPOTLIGHT_POS}.'
    },
    {
        0: 'A {SHAPE} of "{COLOR}" color is visible, positioned at the {YPOS}-{XPOS} of the image, with a spotlight shining from the {SPOTLIGHT_POS}.',
        1: 'A "{COLOR}" {SHAPE} is at the {YPOS}-{XPOS} of the image, illuminated by a light from the {SPOTLIGHT_POS}.',
        2: 'The {YPOS}-{XPOS} of the image shows a "{COLOR}" colored {SHAPE}, highlighted by a light source positioned at the {SPOTLIGHT_POS}.',
        3: 'At the {YPOS}-{XPOS} of the picture is a {SHAPE} in "{COLOR}" color, standing out under a light coming from the {SPOTLIGHT_POS}.',
        4: 'At the {YPOS}-{XPOS} of the image, there is a "{COLOR}" object in the form of a {SHAPE}, with illumination from the {SPOTLIGHT_POS}.'
    },
    {
        0: 'A {SHAPE} of "{COLOR}" color is visible, positioned at the {YPOS}-{XPOS} of the image, under a "{SPOTLIGHT_COLOR}" spotlight shining from the {SPOTLIGHT_POS}.',
        1: 'A "{COLOR}" {SHAPE} is at the {YPOS}-{XPOS} of the image, illuminated by a "{SPOTLIGHT_COLOR}" spotlight from the {SPOTLIGHT_POS}.',
        2: 'The {YPOS}-{XPOS} of the image shows a "{COLOR}" colored {SHAPE}, highlighted by a "{SPOTLIGHT_COLOR}" spotlight positioned at the {SPOTLIGHT_POS}.',
        3: 'At the {YPOS}-{XPOS} of the picture is a {SHAPE} in "{COLOR}" color, standing out under a "{SPOTLIGHT_COLOR}" spotlight coming from the {SPOTLIGHT_POS}.',
        4: 'At the {YPOS}-{XPOS} of the image, there is a "{COLOR}" object in the form of a {SHAPE}, with illumination from a "{SPOTLIGHT_COLOR}" spotlight at the {SPOTLIGHT_POS}.'
    },
    {
        0: 'A {SHAPE} of "{COLOR}" color is visible, positioned at the {YPOS}-{XPOS} of the image, under a "{SPOTLIGHT_COLOR}" spotlight shining from the {SPOTLIGHT_POS}, with a "{BACKGROUND_COLOR}" background.',
        1: 'A "{COLOR}" {SHAPE} is at the {YPOS}-{XPOS} of the image, illuminated by a "{SPOTLIGHT_COLOR}" spotlight from the {SPOTLIGHT_POS}, against a "{BACKGROUND_COLOR}" background.',
        2: 'The {YPOS}-{XPOS} of the image shows a "{COLOR}" colored {SHAPE}, highlighted by a "{SPOTLIGHT_COLOR}" spotlight positioned at the {SPOTLIGHT_POS}, on a "{BACKGROUND_COLOR}" backdrop.',
        3: 'At the {YPOS}-{XPOS} of the picture is a {SHAPE} in "{COLOR}" color, bathed in a "{SPOTLIGHT_COLOR}" spotlight coming from the {SPOTLIGHT_POS}, with a "{BACKGROUND_COLOR}" background.',
        4: 'At the {YPOS}-{XPOS} of the image, there is a "{COLOR}" object in the form of a {SHAPE}, standing out under a "{SPOTLIGHT_COLOR}" spotlight from the {SPOTLIGHT_POS}, against a "{BACKGROUND_COLOR}" background.'
    }
]


def discretize_spotlight_position(value):
    """
    Maps a number (0,1) to one of five directional spotlight positions:
    "northwest", "northeast", "center", "southwest", "southeast".
    
    Parameters:
        value (float): A number in the range (0,1).
    
    Returns:
        str: One of the five fixed spotlight position descriptions.
    """
    if 0.0 <= value < 0.20:
        return "northwest"
    elif 0.20 <= value < 0.40:
        return "northeast"
    elif 0.40 <= value < 0.60:
        return "center"
    elif 0.60 <= value < 0.80:
        return "southwest"
    elif 0.80 <= value <= 1.0:
        return "southeast"
    else:
        return "unknown position"  # Safety check for unexpected values


def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", type=str, required=True)
    args = parser.parse_args()

    # check if directory exists
    assert os.path.exists(args.output_folder)
    
    for mode in ["perturbations", "selections"]:
        for ix in range(7):    
            if mode == "perturbations":
                TEMPLATE = PHRASES[-1]
            else:
                TEMPLATE = PHRASES[ix]
            
            # create output directory
            os.makedirs(os.path.join(args.output_folder, "text", f"{mode}_{ix}"), exist_ok=True)
            output_path = os.path.join(args.output_folder, "text", f"{mode}_{ix}", "text_raw.txt")

            # load latents as dict
            csvpath = os.path.join(args.output_folder, f"latents_text_{mode}_{ix}.csv")
            with open(csvpath, mode='r') as f:
                reader = csv.reader(f, delimiter=",")
                keys = [val for val in next(reader)]  # first row in csv is header
                latents_text = {k: [] for k in keys}
                for row in reader:
                    for k, val in zip(keys, row):
                        try:
                            latents_text[k].append(float(val))
                        except ValueError:  # e.g., when val is a string
                            latents_text[k].append(val)
            num_samples = len(latents_text["object_shape"])

            # generate text
            with open(output_path, 'w') as f:
                for i in range(num_samples):
                    j = int(latents_text["text_phrasing"][i])
                    phrase = TEMPLATE[j]
                    if mode == "perturbations":
                        phrase = phrase.format(
                            SHAPE=SHAPES[latents_text["object_shape"][i]], XPOS=XPOS[latents_text["object_xpos"][i]], YPOS=YPOS[latents_text["object_ypos"][i]],
                            SPOTLIGHT_POS=discretize_spotlight_position(latents_text["spotlight_pos"][i]), COLOR=latents_text["object_color_name"][i],
                            SPOTLIGHT_COLOR=latents_text['splotlight_color_name'][i], BACKGROUND_COLOR=latents_text['background_color_name'][i]
                            )
                    else:
                        if ix == 0:
                            phrase = phrase.format(SHAPE=SHAPES[latents_text["object_shape"][i]])
                        elif ix == 1:
                            phrase = phrase.format(
                                SHAPE=SHAPES[latents_text["object_shape"][i]], XPOS=XPOS[latents_text["object_xpos"][i]]
                                )
                        elif ix == 2:
                            phrase = phrase.format(
                                SHAPE=SHAPES[latents_text["object_shape"][i]], XPOS=XPOS[latents_text["object_xpos"][i]], YPOS=YPOS[latents_text["object_ypos"][i]]
                                )
                        elif ix == 3:
                            phrase = phrase.format(
                                SHAPE=SHAPES[latents_text["object_shape"][i]], XPOS=XPOS[latents_text["object_xpos"][i]], YPOS=YPOS[latents_text["object_ypos"][i]],
                                SPOTLIGHT_POS=discretize_spotlight_position(latents_text["spotlight_pos"][i])
                                )
                        elif ix == 4:
                            phrase = phrase.format(
                                SHAPE=SHAPES[latents_text["object_shape"][i]], XPOS=XPOS[latents_text["object_xpos"][i]], YPOS=YPOS[latents_text["object_ypos"][i]],
                                SPOTLIGHT_POS=discretize_spotlight_position(latents_text["spotlight_pos"][i]), COLOR=latents_text["object_color_name"][i]
                                )
                        elif ix == 5:
                            phrase = phrase.format(
                                SHAPE=SHAPES[latents_text["object_shape"][i]], XPOS=XPOS[latents_text["object_xpos"][i]], YPOS=YPOS[latents_text["object_ypos"][i]],
                                SPOTLIGHT_POS=discretize_spotlight_position(latents_text["spotlight_pos"][i]), COLOR=latents_text["object_color_name"][i],
                                SPOTLIGHT_COLOR=latents_text['splotlight_color_name'][i]
                                )
                        elif ix == 6:
                            phrase = phrase.format(
                                SHAPE=SHAPES[latents_text["object_shape"][i]], XPOS=XPOS[latents_text["object_xpos"][i]], YPOS=YPOS[latents_text["object_ypos"][i]],
                                SPOTLIGHT_POS=discretize_spotlight_position(latents_text["spotlight_pos"][i]), COLOR=latents_text["object_color_name"][i],
                                SPOTLIGHT_COLOR=latents_text['splotlight_color_name'][i], BACKGROUND_COLOR=latents_text['background_color_name'][i]
                                )
                            
                        
                    if i < num_samples - 1:
                        phrase = phrase + "\n"  # newline for all lines except the last
                    f.write(phrase)
            print(f"Done. Saved text to '{output_path}'.")


if __name__ == "__main__":
    
    main()
