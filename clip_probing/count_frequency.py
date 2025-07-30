import os
import csv
from collections import Counter
from tqdm import tqdm
import ahocorasick

# === 📚 Your full concept dictionary ===
concept_dict = {
    'common concepts': {
        'Animal': ['dog', 'cat', 'horse', 'bird', 'elephant', 'giraffe', 'cow', 'zebra', 'rabbit', 'duck'],
        'Clothing': ['shirt', 'pants', 'dress', 'shoes', 'hat', 'jacket', 'skirt', 'tie', 'hoodie', 'socks'],
        'Color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'pink', 'gray', 'brown'],
        'Food': ['pizza', 'burger', 'sandwich', 'salad', 'cake', 'coffee', 'tea', 'beer', 'ice cream', 'noodles'],
        'Object': ['chair', 'table', 'phone', 'laptop', 'car', 'bottle', 'bag', 'cup', 'backpack', 'television'],
        'Role': ['chef', 'teacher', 'athlete', 'doctor', 'engineer', 'artist', 'pilot', 'firefighter', 'police officer', 'lawyer'],
        'Scene': ['beach', 'kitchen', 'forest', 'street', 'park', 'office', 'bedroom', 'classroom', 'stadium', 'playground'],
        'Vehicle': ['bus', 'truck', 'airplane', 'train', 'motorcycle', 'bicycle', 'boat', 'van', 'taxi', 'scooter'],
        'Weather': ['sandstorm', 'drought', 'rain', 'snow', 'fog', 'rainbow', 'storm', 'moonlight', 'overcast', 'frost']
    },
    'valuable under-captioned concepts': {
        'Texture': ['glossy', 'matte', 'rough', 'smooth', 'fuzzy', 'silky', 'grainy', 'wrinkled', 'slippery', 'furry'],
        'POV': ['close-up', 'wide shot', 'top-down', 'low angle', 'high angle', 'aerial', 'shallow-depth', 'long exposure'],
        'Emot.': ['tired', 'focused', 'surprised', 'proud', 'shy', 'bored', 'confused', 'excited', 'thoughtful', 'nervous']
    },
    'nuisance cues': {
        'Postproc.': ['hdr', 'desaturated', 'color-filtered', 'over-sharpened', 'saturated', 'sepia', 'heavily edited', 'bokeh', 'tilt-shift'],
    },
    'sensitive concepts': {
        'Trait': ['easy-going', 'cold-hearted', 'evil', 'criminal', 'immoral', 'pure', 'corrupt', 'hero', 'cheater', 'deserving'],
        'Stere.': ['exotic look', 'ethnic look', 'slacker', 'aggressive look', 'gangster look', 'terrorist look', 'illegal look', 'privileged', 'ghetto look']
    }
}
concept_synonyms = {
    # ===== COMMON CONCEPTS =====
    # Animals
    "dog":            ["dog", "puppy", "canine"],
    "cat":            ["cat", "kitten", "feline"],
    "horse":          ["horse", "pony", "stallion"],
    "bird":           ["bird", "parrot", "sparrow", "avian"],
    "elephant":       ["elephant"],
    "giraffe":        ["giraffe"],
    "cow":            ["cow", "cattle", "bull", "calf"],
    "zebra":          ["zebra"],
    "rabbit":         ["rabbit", "bunny", "hare"],
    "duck":           ["duck", "duckling"],

    # Clothing
    "shirt":          ["shirt", "t-shirt", "tee", "blouse"],
    "pants":          ["pants", "trousers", "jeans", "slacks"],
    "dress":          ["dress", "gown"],
    "shoes":          ["shoes", "sneakers", "boots", "footwear"],
    "hat":            ["hat", "cap", "beanie"],
    "jacket":         ["jacket", "coat", "parka"],
    "skirt":          ["skirt"],
    "tie":            ["tie", "necktie"],
    "hoodie":         ["hoodie", "hooded sweatshirt"],
    "socks":          ["socks", "stockings"],

    # Colors
    "red":            ["red", "crimson"],
    "blue":           ["blue", "azure"],
    "green":          ["green", "emerald"],
    "yellow":         ["yellow", "golden"],
    "black":          ["black", "dark"],
    "white":          ["white", "ivory"],
    "orange":         ["orange"],
    "pink":           ["pink", "rose"],
    "gray":           ["gray", "grey", "ash"],
    "brown":          ["brown", "chocolate"],

    # Food
    "pizza":          ["pizza", "slice"],
    "burger":         ["burger", "hamburger", "cheeseburger"],
    "sandwich":       ["sandwich", "sub"],
    "salad":          ["salad", "greens"],
    "cake":           ["cake", "pastry"],
    "coffee":         ["coffee", "espresso", "latte", "cappuccino"],
    "tea":            ["tea", "chai"],
    "beer":           ["beer", "ale", "lager"],
    "ice cream":      ["ice cream", "gelato", "sundae"],
    "noodles":        ["noodles", "pasta", "spaghetti"],

    # Objects
    "chair":          ["chair", "seat", "armchair"],
    "table":          ["table", "desk"],
    "phone":          ["phone", "cellphone", "mobile", "smartphone"],
    "laptop":         ["laptop", "notebook", "computer"],
    "car":            ["car", "automobile", "vehicle"],
    "bottle":         ["bottle", "flask", "jug"],
    "bag":            ["bag", "handbag", "purse", "tote"],
    "cup":            ["cup", "mug", "glass"],
    "backpack":       ["backpack", "knapsack", "pack"],
    "television":     ["television", "tv", "screen"],

    # Roles
    "chef":           ["chef", "cook"],
    "teacher":        ["teacher", "instructor", "professor"],
    "athlete":        ["athlete", "sportsman", "sportsperson"],
    "doctor":         ["doctor", "physician", "medic"],
    "engineer":       ["engineer", "technician"],
    "artist":         ["artist", "painter", "illustrator"],
    "pilot":          ["pilot", "aviator"],
    "firefighter":    ["firefighter", "fireman"],
    "police officer": ["police officer", "cop", "policeman"],
    "lawyer":         ["lawyer", "attorney"],

    # Scenes
    "beach":          ["beach", "seashore", "coast"],
    "kitchen":        ["kitchen", "cooking area"],
    "forest":         ["forest", "woods", "jungle"],
    "street":         ["street", "road", "avenue", "sidewalk"],
    "park":           ["park", "playground", "green area"],
    "office":         ["office", "workplace"],
    "bedroom":        ["bedroom", "sleeping room"],
    "classroom":      ["classroom", "lecture room"],
    "stadium":        ["stadium", "arena"],
    "playground":     ["playground", "play area"],

    # Vehicles
    "bus":            ["bus", "coach"],
    "truck":          ["truck", "lorry"],
    "airplane":       ["airplane", "plane", "jet", "aircraft"],
    "train":          ["train", "locomotive"],
    "motorcycle":     ["motorcycle", "motorbike", "bike"],
    "bicycle":        ["bicycle", "bike", "cycle"],
    "boat":           ["boat", "ship", "vessel"],
    "van":            ["van", "minivan"],
    "taxi":           ["taxi", "cab"],
    "scooter":        ["scooter", "moped"],

    # Weather
    "sandstorm":      ["sandstorm", "dust storm"],
    "drought":        ["drought", "dry spell"],
    "rain":           ["rain", "rainy", "drizzle"],
    "snow":           ["snow", "snowfall"],
    "fog":            ["fog", "mist"],
    "rainbow":        ["rainbow"],
    "storm":          ["storm", "thunderstorm", "tempest"],
    "moonlight":      ["moonlight", "moonlit"],
    "overcast":       ["overcast", "cloudy"],
    "frost":          ["frost", "frozen", "ice"],

    # ===== VALUABLE UNDER-CAPTIONED CONCEPTS =====
    # Texture
    "glossy":         ["glossy", "shiny"],
    "matte":          ["matte", "dull"],
    "rough":          ["rough", "coarse"],
    "smooth":         ["smooth", "even", "sleek"],
    "fuzzy":          ["fuzzy", "fluffy"],
    "silky":          ["silky", "silken", "smooth"],
    "grainy":         ["grainy", "gritty"],
    "wrinkled":       ["wrinkled", "creased"],
    "slippery":       ["slippery", "slick"],
    "furry":          ["furry", "hairy"],

    # Viewpoint
    "close-up":       ["close-up", "macro", "zoomed in"],
    "wide shot":      ["wide shot", "panoramic"],
    "top-down":       ["top-down", "bird's-eye view"],
    "low angle":      ["low angle", "from below"],
    "high angle":     ["high angle", "from above"],
    "aerial":         ["aerial", "from the air", "drone view"],
    "shallow-depth":  ["shallow-depth", "bokeh"],
    "long exposure":  ["long exposure", "light trail"],

    # Mental State
    "tired":          ["tired", "exhausted", "sleepy"],
    "focused":        ["focused", "concentrating", "attentive"],
    "surprised":      ["surprised", "shocked", "astonished"],
    "proud":          ["proud", "confident", "self-satisfied"],
    "shy":            ["shy", "timid", "bashful"],
    "bored":          ["bored", "uninterested"],
    "confused":       ["confused", "perplexed"],
    "excited":        ["excited", "thrilled", "eager"],
    "thoughtful":     ["thoughtful", "pensive", "reflective"],
    "nervous":        ["nervous", "anxious", "uneasy"],

    # ===== NUISANCE CUES =====
    "hdr":                ["hdr", "high dynamic range"],
    "desaturated":        ["desaturated", "faded", "muted"],
    "color-filtered":     ["color-filtered", "colorized", "tinted"],
    "over-sharpened":     ["over-sharpened", "too sharp"],
    "saturated":          ["saturated", "vivid", "intense color"],
    "sepia":              ["sepia", "brownish", "sepia-toned"],
    "heavily edited":     ["heavily edited", "over-processed"],
    "bokeh":              ["bokeh", "blurry background"],
    "tilt-shift":         ["tilt-shift", "miniature effect"],

    # ===== SENSITIVE CONCEPTS =====
    # Moral Bias
    "easy-going":         ["easy-going", "laid-back", "relaxed"],
    "cold-hearted":       ["cold-hearted", "unfeeling", "heartless"],
    "evil":               ["evil", "wicked", "malicious"],
    "criminal":           ["criminal", "offender", "lawbreaker"],
    "immoral":            ["immoral", "unethical"],
    "pure":               ["pure", "innocent"],
    "corrupt":            ["corrupt", "dishonest"],
    "hero":               ["hero", "savior", "champion"],
    "cheater":            ["cheater", "deceiver", "fraud"],
    "deserving":          ["deserving", "worthy"],

    # Stereotypes
    "exotic look":        ["exotic look", "foreign appearance"],
    "ethnic look":        ["ethnic look", "ethnic appearance"],
    "slacker":            ["slacker", "lazy person"],
    "aggressive look":    ["aggressive look", "hostile appearance"],
    "delinquent":         ["delinquent", "troublemaker", "lawbreaker", "offender", "juvenile delinquent"],
    "terrorist look":     ["terrorist look", "suspicious appearance"],
    "illegal look":       ["illegal look", "unauthorized appearance"],
    "privileged":         ["privileged", "advantaged"],
    "ghetto look":        ["ghetto look", "rough appearance"]
}


def flatten_concepts(concept_dict):
    """Returns sorted canonical concept list (lowercase)."""
    all_concepts = set()
    for group in concept_dict.values():
        for sublist in group.values():
            all_concepts.update([c.lower() for c in sublist])
    return sorted(all_concepts)

def build_synonym_map(concept_synonyms):
    """Returns:
        - all_synonyms: set of all lowercase synonyms
        - synonym2concept: maps synonym -> canonical concept (lowercase)
    """
    all_synonyms = set()
    synonym2concept = dict()
    for concept, syns in concept_synonyms.items():
        canon = concept.lower()
        for syn in syns:
            s = syn.lower()
            all_synonyms.add(s)
            synonym2concept[s] = canon
    return sorted(all_synonyms, key=len, reverse=True), synonym2concept

def build_aho_automaton(synonyms):
    A = ahocorasick.Automaton()
    for idx, syn in enumerate(synonyms):
        A.add_word(syn, (idx, syn))
    A.make_automaton()
    return A

def count_with_synonyms_aho(caption_dir, concept_synonyms):
    all_synonyms, synonym2concept = build_synonym_map(concept_synonyms)
    automaton = build_aho_automaton(all_synonyms)
    concept_caption_counts = Counter()
    total_caption_count = 0

    files = sorted([f for f in os.listdir(caption_dir) if f.endswith('.txt')])
    for fname in tqdm(files, desc="Counting with Synonyms + Aho-Corasick"):
        with open(os.path.join(caption_dir, fname), 'r', encoding='utf-8') as f:
            for line in f:
                caption = line.strip().lower()
                if not caption:
                    continue
                total_caption_count += 1
                matched_concepts = set()
                for end_idx, (idx, found_syn) in automaton.iter(caption):
                    concept = synonym2concept[found_syn]
                    matched_concepts.add(concept)
                for c in matched_concepts:
                    concept_caption_counts[c] += 1

    return concept_caption_counts, total_caption_count

def save_frequencies_to_csv(concept_counts, total_captions, concept_groups, output_csv="clip_probing/concept_frequencies.csv"):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Concept", "Count", "Percentage"])
        for concept in flatten_concepts(concept_groups):
            count = concept_counts.get(concept.lower(), 0)
            percentage = count / total_captions * 100 if total_captions > 0 else 0
            writer.writerow([concept, count, f"{percentage:.4f}"])
        
if __name__ == '__main__':
    concept_caption_counts, total_caption_count = count_with_synonyms_aho(
        "laion400m-metadata/captions", concept_synonyms)
    save_frequencies_to_csv(concept_caption_counts, total_caption_count, concept_dict)
    print("Done Done London.")