class DebugObserver:
    def update(self, event_type, data):
        print(f"[DEBUG] {event_type}: {data}")

class DataGenerator:
    def __init__(self):
        self.observers = []

    def attach(self, observer):
        self.observers.append(observer)

    def notify(self, event_type, data):
        for observer in self.observers:
            observer.update(event_type, data)

    def generate(self):
        # Beispiel: Generierung eines Bildes
        self.notify("START_GENERATION", {"image_id": 1})
        # Bild generieren ...
        self.notify("END_GENERATION", {"image_id": 1})