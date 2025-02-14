import numpy as np
from PIL import Image, ImageDraw
import math


class PuzzlePiece:
    def __init__(self, bounds, path, mask):
        self.bounds = bounds  # (x1, y1, x2, y2)
        self.path = path  # List of points defining the piece shape
        self.mask = mask  # PIL Image mask for the piece


class ImagePuzzleGenerator:
    def __init__(self, image_path: str, rows: int, cols: int):
        # Load and process the original image
        self.original_image = Image.open(image_path).convert("RGBA")
        self.width = self.original_image.width
        self.height = self.original_image.height
        self.rows = rows
        self.cols = cols

        # Calculate piece dimensions
        self.piece_width = self.width / cols
        self.piece_height = self.height / rows

        # Size of tabs (20% of smaller piece dimension)
        self.tab_size = min(self.piece_width, self.piece_height) * 0.2

        # Tab/slot shapes: dict to share edges between pieces
        self.edge_shapes = {}

    def _create_tab_points(
        self, start: tuple, end: tuple, is_tab: bool, orientation: str
    ) -> list:
        """
        Create consistent points for a puzzle tab or slot between two points.
        """
        sx, sy = start
        ex, ey = end

        # Calculate midpoint and perpendicular direction
        mid_x = (sx + ex) / 2
        mid_y = (sy + ey) / 2
        dx = ex - sx
        dy = ey - sy
        length = math.sqrt(dx * dx + dy * dy)
        unit_perp_x = -dy / length
        unit_perp_y = dx / length

        # Direction: tab or slot
        direction = 1 if is_tab else -1

        # Flip for certain orientations
        if orientation in {"left", "top"}:
            direction *= -1

        # Control point for Bézier curve
        ctrl_dist = self.tab_size * direction
        c1x = mid_x + unit_perp_x * ctrl_dist
        c1y = mid_y + unit_perp_y * ctrl_dist

        # Bézier curve points
        steps = 10
        points = [
            (
                (1 - t) ** 2 * sx + 2 * (1 - t) * t * c1x + t ** 2 * ex,
                (1 - t) ** 2 * sy + 2 * (1 - t) * t * c1y + t ** 2 * ey,
            )
            for t in np.linspace(0, 1, steps + 1)
        ]
        return points

    def generate_piece_path(self, row: int, col: int) -> list:
        """
        Generate the path for a single puzzle piece, ensuring edges match neighboring pieces.
        """
        x1 = col * self.piece_width
        y1 = row * self.piece_height
        x2 = x1 + self.piece_width
        y2 = y1 + self.piece_height

        path = []

        # Top edge
        if row == 0:
            path.extend([(x1, y1), (x2, y1)])  # Flat edge
        else:
            key = f"{row - 1},{col}-bottom"
            is_tab = self.edge_shapes[key]
            path.extend(self._create_tab_points((x1, y1), (x2, y1), not is_tab, "top"))

        # Right edge
        if col == self.cols - 1:
            path.extend([(x2, y1), (x2, y2)])  # Flat edge
        else:
            key = f"{row},{col}-right"
            is_tab = np.random.choice([True, False]) if key not in self.edge_shapes else self.edge_shapes[key]
            self.edge_shapes[key] = is_tab
            path.extend(self._create_tab_points((x2, y1), (x2, y2), is_tab, "right"))

        # Bottom edge
        if row == self.rows - 1:
            path.extend([(x2, y2), (x1, y2)])  # Flat edge
        else:
            key = f"{row},{col}-bottom"
            is_tab = np.random.choice([True, False]) if key not in self.edge_shapes else self.edge_shapes[key]
            self.edge_shapes[key] = is_tab
            path.extend(
                reversed(
                    self._create_tab_points((x1, y2), (x2, y2), is_tab, "bottom")
                )
            )

        # Left edge
        if col == 0:
            path.extend([(x1, y2), (x1, y1)])  # Flat edge
        else:
            key = f"{row},{col - 1}-right"
            is_tab = self.edge_shapes[key]
            path.extend(
                reversed(self._create_tab_points((x1, y1), (x1, y2), not is_tab, "left"))
            )

        return path

    def create_piece_mask(self, path: list) -> Image.Image:
        """Create a mask for the puzzle piece."""
        mask = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(path, fill=255)
        return mask

    def generate_puzzle(self) -> list:
        """Generate all puzzle pieces from the image."""
        puzzle_pieces = []

        # Generate each piece
        for row in range(self.rows):
            for col in range(self.cols):
                # Generate the piece path
                path = self.generate_piece_path(row, col)

                # Create mask for this piece
                mask = self.create_piece_mask(path)

                # Crop the piece
                piece_img = self.original_image.copy()
                piece_img.putalpha(mask)
                bounds = (
                    int(col * self.piece_width - self.tab_size),
                    int(row * self.piece_height - self.tab_size),
                    int((col + 1) * self.piece_width + self.tab_size),
                    int((row + 1) * self.piece_height + self.tab_size),
                )
                piece_img = piece_img.crop(bounds)

                # Save each piece
                piece_path = f"puzzle_piece_{row}_{col}.png"
                piece_img.save(piece_path)
                print(f"Saved piece: {piece_path}")

                puzzle_pieces.append(piece_img)

        return puzzle_pieces


def main():
    # Create puzzle from the input image
    generator = ImagePuzzleGenerator("butterfly.png", rows=3, cols=4)
    pieces = generator.generate_puzzle()

    # Display puzzle layout
    canvas = Image.new("RGBA", (generator.width, generator.height), (255, 255, 255, 0))
    for i, piece in enumerate(pieces):
        row = i // generator.cols
        col = i % generator.cols
        x = int(col * generator.piece_width)
        y = int(row * generator.piece_height)
        canvas.paste(piece, (x, y), piece)

    # Save the result
    canvas.save("puzzle_complete.png")


if __name__ == "__main__":
    main()
