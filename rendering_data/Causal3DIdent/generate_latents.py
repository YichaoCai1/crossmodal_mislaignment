"""Generate latents for the Multimodal3DIdent dataset."""

import argparse
import colorsys
import os
import random

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import CSS4_COLORS, TABLEAU_COLORS, XKCD_COLORS


class ColorPalette(object):
    """Color palette with utility functions."""

    def __init__(self, palette):
        """
        Initialize color palette.

        Args:
            palette (dict): dictionary of color-names to hex-values. For
                example: {"my_blue": "#0057b7", "my_yellow": "#ffd700"}.
        """
        # precompute rgb to name/hex for all keys and values in the palette
        self.rgb_to_name = {self.hex_to_rgb(v): k for k, v in palette.items()}
        self.rgb_to_hex = {self.hex_to_rgb(v): v for k, v in palette.items()}
        self.palette = palette

    def nearest_neighbor(self, rgb_value, return_name=True):
        """Given an rgb-value, find the nearest neighbor among the values of the palette."""
        assert len(rgb_value) == 3
        rgb_value_arr = np.array(rgb_value)
        min_dist = np.inf  # minimal distance
        rgb_nn = None      # rgb-value of the nearest neighbor
        for rgb_key in self.rgb_to_name.keys():
            dist = np.linalg.norm(np.array(rgb_key) - rgb_value_arr)  # euclidian distance
            if dist < min_dist:
                min_dist = dist
                rgb_nn = rgb_key
        if return_name:
            return self.rgb_to_name[rgb_nn]
        else:
            return rgb_nn

    @staticmethod
    def hex_to_rgb(hex_value):
        """Transform hex-code "#rrggbb" to rgb-tuple (r, g, b)."""
        rgb_value = matplotlib.colors.to_rgb(hex_value)
        return rgb_value

    @staticmethod
    def hue_to_rgb(hue_value):
        """Transform hue-value (between 0 and 1) to rgb-tuple (r, g, b)."""
        rgb_value = colorsys.hsv_to_rgb(hue_value, 1.0, 1.0)  # s and v are constant
        return rgb_value


def hue_to_colorname(object_hue, color_palettes):
    """Map hue value to a matching color name from a randomly sampled color palette.

    Args:
        object_hue (list or np.array): hue values in the interval [0, 1].
        color_palettes (list): list of color palettes (class `ColorPalette`).

    Returns:
        List of color names as an np.array of strings.
    """
    object_colorname = []
    for h in object_hue:
        j = np.random.randint(len(color_palettes))
        cp = color_palettes[j]
        rgb = cp.hue_to_rgb(h)
        colorname = cp.nearest_neighbor(rgb)  # color name of nearest neighbor
        object_colorname.append(colorname)
    return np.array(object_colorname)


def hue_to_colorname(object_hue, color_palette):
    """Map hue value to a matching color name from a assigned sampled color palette.

    Args:
        object_hue (list or np.array): hue values in the interval [0, 1].
        color_palette : assigned color palette.

    Returns:
        List of color names as an np.array of strings.
    """
    object_colorname = []
    for h in object_hue:
        rgb = color_palette.hue_to_rgb(h)
        colorname = color_palette.nearest_neighbor(rgb)  # color name of nearest neighbor
        object_colorname.append(colorname)
    return np.array(object_colorname)


def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--n-points", type=int, required=True)
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args()

    # print args
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    # set all seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # create output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # define color palettes
    color_palettes = \
        [ColorPalette(x) for x in (TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS)]
        
    color_names = [[], [], []]
    unique_names = [[], [], []]
    color_indices = [[], [], []]
    colorname_to_index = [[], [], []]
    for i, cp in enumerate(color_palettes):
        for k in cp.palette.keys():
            color_names[i].append(k)
        color_names[i] = sorted(color_names[i])  # sort names to ensure the same order
    
        unique_names[i], color_indices[i] = np.unique(color_names[i], return_inverse=True)
        colorname_to_index[i] = \
            {name: index for (name, index) in zip(color_names[i], color_indices[i])}


    # define latent space
    # -------------------
    selections = [
        ["object_shape"],
        ["object_shape", "object_xpos"],
        ["object_shape", "object_xpos", "object_ypos"],
        ["object_shape", "object_xpos", "object_ypos", "spotlight_pos"],
        ["object_shape", "object_xpos", "object_ypos", "spotlight_pos", "object_color_name", "object_color_index"],
        ["object_shape", "object_xpos", "object_ypos", "spotlight_pos", "object_color_name", "object_color_index",\
            "splotlight_color_name", "splotlight_color_index"],
        ["object_shape", "object_xpos", "object_ypos", "spotlight_pos", "object_color_name", "object_color_index",\
            "splotlight_color_name", "splotlight_color_index", "background_color_name", "background_color_index"],
    ]
    
    perturbations = [
        [],
        ["background_color"],
        ["splotlight_color", "background_color"],
        ["object_color", "splotlight_color", "background_color"],
        ["spotlight_pos", "object_color", "splotlight_color", "background_color"],
        ["object_ypos", "spotlight_pos", "object_color", "splotlight_color", "background_color"],
        ["object_xpos", "object_ypos", "spotlight_pos", "object_color", "splotlight_color", "background_color"]
    ]
    
    biases = {"selections": selections, "perturbations": perturbations}
    
    # image latent
    object_xpos = np.random.randint(0, 3, args.n_points)    # discrete, 7 values drawn uniformly
    object_ypos = np.random.randint(0, 3, args.n_points)    # discrete, 3 values drawn uniformly
    object_shape = np.random.randint(0, 7, args.n_points)   # discrete, 3 values drawn uniformly
    spotlight_pos = np.random.rand(args.n_points)           # continuous, uniform on [0, 1]
    
    spotlight_hue = (spotlight_pos + np.random.rand(args.n_points)) /2.           # spotlight_pos -> splotlight_hue
    # spotlight_pos -> background_hue, splotlight_hue -> background_hue
    background_hue = (spotlight_pos + spotlight_hue + np.random.rand(args.n_points)) / 3.    
    
    object_hue = ((object_xpos + object_ypos)/2. + np.random.rand(args.n_points)) / 3.     # object_xpos, object_ypos -> object_hue
    
    latents_image = {
        "object_shape": object_shape,  
        "object_xpos": object_xpos,                               
        "object_ypos": object_ypos,    
        "object_zpos": np.zeros(args.n_points),                   # constant
        "object_alpharot": np.random.rand(args.n_points),         # continuous, uniform on [0, 1]
        "object_betarot": np.random.rand(args.n_points),          # continuous, uniform on [0, 1]
        "object_gammarot": np.random.rand(args.n_points),         # continuous, uniform on [0, 1]
        "object_color": object_hue,                               
        "spotlight_pos": spotlight_pos,          
        "spotlight_color": spotlight_hue,         
        "background_color": background_hue,       
    }
    
    pd.DataFrame(latents_image).to_csv(os.path.join(args.output_folder, "latents_image.csv"), index=False)                
    
    for bias_type in biases:
        bias_settings = biases[bias_type]            
        if bias_type == "selections":
            # text latents
            object_colorname = hue_to_colorname(object_hue, color_palettes[0])  # use distinct color for each semanticss
            object_colorindex = np.array([colorname_to_index[0][cname] for cname in object_colorname])
            
            spotlight_colorname = hue_to_colorname(spotlight_hue, color_palettes[1])
            spotlight_colorindex = np.array([colorname_to_index[1][cname] for cname in spotlight_colorname])
            
            background_colorname = hue_to_colorname(background_hue, color_palettes[2])
            background_colorindex = np.array([colorname_to_index[2][cname] for cname in background_colorname])
            
            latents_text = {
                "object_shape": latents_image["object_shape"],            # discrete, 7 values drawn uniformly
                "object_xpos": latents_image["object_xpos"],              # discrete, 3 values drawn uniformly
                "object_ypos": latents_image["object_ypos"],              # discrete, 3 values drawn uniformly
                "spotlight_pos": latents_image["spotlight_pos"], 
                "object_color_name": object_colorname,                    # discrete, color name as string
                "object_color_index": object_colorindex,                  # discrete, color name as unique integer
                "splotlight_color_name": spotlight_colorname, 
                "splotlight_color_index": spotlight_colorindex,
                "background_color_name": background_colorname,
                "background_color_index": background_colorindex,
                "text_phrasing": np.random.randint(0, 5, args.n_points),  # discrete, 5 values drawn uniformly
            }
            for i, bias_list in enumerate(bias_settings):
                selected_latents = {}
                for k in latents_text:
                    if k in bias_list:
                        selected_latents[k] = latents_text[k]    
                selected_latents["text_phrasing"] = latents_text[k] 
                
                # save latents to disk as csv files
                pd.DataFrame(selected_latents).to_csv(os.path.join(args.output_folder, f"latents_text_{bias_type}_{i}.csv"), index=False)
                print(f"\nDone. Saved latents to '{args.output_folder}/'.")
                        
        else:          
            for i, bias_list in enumerate(bias_settings):
                t_background_hue = np.random.rand(args.n_points)  if "background_color" in bias_list else object_hue
                t_spotlight_hue = np.random.rand(args.n_points) if "splotlight_color" in bias_list else spotlight_hue
                t_object_hue = np.random.rand(args.n_points) if "object_color" in bias_list else object_hue
                t_spotlight_pos =  np.random.rand(args.n_points) if "spotlight_pos" in bias_list else spotlight_pos
                t_object_ypos = np.random.randint(0, 3, args.n_points) if "object_ypos" in bias_list else object_ypos
                t_object_xpos = np.random.randint(0, 3, args.n_points) if "object_xpos" in bias_list else object_xpos
                
                object_colorname = hue_to_colorname(t_object_hue, color_palettes[0])  # use distinct color for each semanticss
                object_colorindex = np.array([colorname_to_index[0][cname] for cname in object_colorname])
                
                spotlight_colorname = hue_to_colorname(t_spotlight_hue, color_palettes[1])
                spotlight_colorindex = np.array([colorname_to_index[1][cname] for cname in spotlight_colorname])
                
                background_colorname = hue_to_colorname(t_background_hue, color_palettes[2])
                background_colorindex = np.array([colorname_to_index[2][cname] for cname in background_colorname])
                
                latents_text = {
                    "object_shape": latents_image["object_shape"],            # discrete, 7 values drawn uniformly
                    "object_xpos": t_object_xpos,              # discrete, 3 values drawn uniformly
                    "object_ypos": t_object_ypos,              # discrete, 3 values drawn uniformly
                    "spotlight_pos": t_spotlight_pos, 
                    "object_color_name": object_colorname,                    # discrete, color name as string
                    "object_color_index": object_colorindex,                  # discrete, color name as unique integer
                    "splotlight_color_name": spotlight_colorname,
                    "splotlight_color_index": spotlight_colorindex,
                    "background_color_name": background_colorname,
                    "background_color_index": background_colorindex,
                    "text_phrasing": np.random.randint(0, 5, args.n_points),  # discrete, 5 values drawn uniformly
                }
                # save latents to disk as csv files
                pd.DataFrame(latents_text).to_csv(os.path.join(args.output_folder, f"latents_text_{bias_type}_{i}.csv"), index=False)
                print(f"\nDone. Saved latents to '{args.output_folder}/'.")


if __name__ == "__main__":
    main()
