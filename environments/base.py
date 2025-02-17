import heapq
import shutil
import textwrap
import numpy as np
import scipy
import gymnasium as gym
import math


class EventQueue:
    def __init__(self):
        self.events = []

    def peek(self):
        return self.events[0]

    def add_event(self, event):
        heapq.heappush(self.events, event)

    def cancel_event(self, event):
        event.cancel()

    def pop_event(self):
        while self.events:
            event = heapq.heappop(self.events)
            if not event.canceled:
                return event
        raise IndexError("No more events in the queue")

    def reset(self):
        self.events = []


class Event:

    def __init__(self, event_type, handler, time, data=None):
        self.type = event_type
        self.handler = handler
        self.time = time
        self.data = data
        self.canceled = False

    def __repr__(self):
        return f"(Event.{self.type}, handler={self.handler}, time={self.time})"

    def __lt__(self, other_event):
        return self.time < other_event.time  # Allows comparison in the priority queue

    def process(self):
        self.handler.handle_event(self)

    def cancel(self):
        self.canceled = True


class DiscreteEventEnvironment(gym.Env):
    def __init__(self, time_limit, verbose=False):
        self.time_limit = time_limit
        self.verbose = verbose
        self.entities = set()
        self.action_space = None
        self.observation_space = None

    def __str__(self):
        return "Env"

    def step(self, action):
        pass

    def reset(
        self,
        seed = None,
        options = None # additional information to specify how the environment is reset
    ):
        super().reset(seed=seed)
        self.time = 0
        self.previous_event_time = 0
        self.future_events = EventQueue()
        for entity in self.entities:
            entity.reset()

    def update_state(self):
        for entity in self.entities:
            entity.update_state()

    def to_dict(self):
        pass

    def log(self, entity, message, category="INFO"):
        if self.verbose:
            # Get the terminal width
            terminal_width = shutil.get_terminal_size().columns

            # Calculate the starting position for the message column
            prefix_width = 15 + 3 + 10 + 3 + 10 + 3  # Width of time, category, entity, and dividers
            message_width = max(terminal_width - prefix_width, 20)  # Ensure a minimum message width

            # Wrap the message
            wrapped_message = textwrap.wrap(message, width=message_width)

            # Print the first line
            print(f"{self.time:>15.4f} | {category:>10} | {str(entity):>10} | {wrapped_message[0]}")

            # Print the subsequent lines with proper alignment
            for line in wrapped_message[1:]:
                print(f"{'':>15} | {'':>10} | {'':>10} | {line}")
            print("\n")

    def add_entity(self, entity):
        entity.initialize(self)
        self.entities.add(entity)

    def remove_entity(self, entity):
        self.entities.remove(entity)

    def schedule_event(self, event):
        self.future_events.add_event(event)

    def cancel_event(self, event):
        self.future_events.cancel_event(event)

    
class EventHandler():
    def __init__(self):
        self.environment = None
        
    # This method should called when the handler is added to the environment
    def initialize(self, environment):
        self.environment = environment

    def schedule_event(self, event):
        self.environment.schedule_event(event)
        self.log(f"Scheduled {event}")

    def cancel_event(self, event):
        self.environment.cancel_event(event)
        self.log(f"Canceled {event}")

    def handle_event(self, event):
        pass

    def log(self, message, category="INFO"):
        self.environment.log(self, message, category)


class Entity(EventHandler):

    def __init__(self, entity_id, entity_type):
        super().__init__()
        self.id = entity_id
        self.type = entity_type
    
    def __hash__(self):
        return hash((self.id, self.type))

    def __eq__(self, other):
        return self.type == other.type and self.id == other.id

    def reset(self):
        pass

    def to_dict(self):
        pass

    def get_observation(self):
        pass

    def update_state(self):
        pass

class RandomVariateGenerator:

    def __init__(self, distribution, parameters, library="numpy"):
        self.rng = None
        self.distribution = distribution # name of numpy distribution
        self.parameters = parameters
        self.library = library

    def reset(self, rng):
        self.rng = rng

    def generate(self):
        if self.library == "numpy":
            return self.rng.__getattribute__(self.distribution)(**self.parameters)
        elif self.library == "scipy":
            return scipy.stats.__getattribute__(self.distribution).rvs(**self.parameters, random_state=self.rng)

    
class Location():
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def __repr__(self):
        return f"({self.latitude}, {self.longitude})"

    def to_dict(self):
        return {
            "lat": self.latitude,
            "lon": self.longitude
        }

    def copy(self):
        return Location(self.latitude, self.longitude)
    
    def distance_to(self, destination):
        """ Calculate the distance to a destination """

        # Convert degrees to radians
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(destination.latitude), math.radians(destination.longitude)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        R = 6371000 # Radius of Earth in meters
        distance = R * c

        return distance

    def initial_bearing_to(self, destination):
        """ Calculate the initial bearing from the current location to the destination """
        lat1 = math.radians(self.latitude)
        lon1 = math.radians(self.longitude)
        lat2 = math.radians(destination.latitude)
        lon2 = math.radians(destination.longitude)

        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.atan2(x, y)
        return bearing

    def update(self, distance, destination):
        """ Update the location by moving along a bearing for a given distance """
        R = 6371000  # Earth's radius in meters
        lat1 = math.radians(self.latitude)
        lon1 = math.radians(self.longitude)
        bearing = self.initial_bearing_to(destination)

        # Calculate new latitude
        new_lat = math.asin(math.sin(lat1) * math.cos(distance / R) +
                            math.cos(lat1) * math.sin(distance / R) * math.cos(bearing))

        # Calculate new longitude
        new_lon = lon1 + math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat1),
                                    math.cos(distance / R) - math.sin(lat1) * math.sin(new_lat))

        # Convert back to degrees
        self.latitude = math.degrees(new_lat)
        self.longitude = math.degrees(new_lon)