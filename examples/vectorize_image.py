#!/usr/bin/env python3
"""Example script for vectorizing an image"""

import torch
from PIL import Image
import torchvision.transforms as T
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description='Vectorize an image using hierarchical diffusion curves')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--solver', choices=['torch', 'scipy'], default='torch',
                        help='Solver backend to use')
    parser.add_argument('--levels', type=int, default=3,
                        help='Number of pyramid levels')
    args = parser.parse_args()

    # Load image
    image = Image.open(args.input).convert('RGB')
    transform = T.Compose([
        T.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Vectorize
    pipeline = VectorizationPipeline(solver_type=args.solver, num_levels=args.levels)
    result = pipeline.vectorize(image_tensor)

    # Save reconstruction
    reconstruction = result['reconstruction'][0]
    reconstruction = reconstruction.clamp(0, 1)
    output_image = T.ToPILImage()(reconstruction)
    output_image.save(args.output)

    print(f"Vectorization complete. Found {len(result['curves'])} curves.")
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
