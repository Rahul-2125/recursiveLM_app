from typing import List, Dict, Any


class DataProcessor:
    """Process and transform data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processed_count = 0

    def process(self, items: List[Dict]) -> List[Dict]:
        """Process a list of items and return transformed results."""
        results = []
        for item in items:
            transformed = self._transform(item)
            results.append(transformed)
            self.processed_count += 1
        return results

    def _transform(self, item: Dict) -> Dict:
        """Apply transformations to a single item."""
        return {
            "id": item.get("id"),
            "name": item.get("name", "").upper(),
            "value": item.get("value", 0) * self.config.get("multiplier", 1),
            "processed": True,
        }

    def get_stats(self) -> Dict[str, int]:
        """Return processing statistics."""
        return {
            "processed_count": self.processed_count,
            "config_keys": len(self.config),
        }


def main():
    config = {"multiplier": 2, "debug": True}
    processor = DataProcessor(config)

    data = [
        {"id": 1, "name": "item1", "value": 10},
        {"id": 2, "name": "item2", "value": 20},
    ]

    results = processor.process(data)
    print(f"Results: {results}")
    print(f"Stats: {processor.get_stats()}")


if __name__ == "__main__":
    main()
