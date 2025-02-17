from enum import Enum, IntEnum
import numpy as np
from collections import defaultdict
import math
import csv
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.gaussian_process
from .base import *
from . import utilities


class EventType(Enum):
    RECEIVED_REQUEST = 1
    RECEIVED_DECISION = 2
    COMPLETED_FLIGHT_PREPARATIONS = 3
    ARRIVED_AT_PICKUP_SITE = 4
    LOADED_PATIENTS = 5
    ARRIVED_AT_MTF = 6
    OFFLOADED_PATIENTS = 7
    ARRIVED_AT_STAGING_AREA = 8
    COMPLETED_REFUELING = 9
    COMPLETED_PATIENT_TREATMENT = 10


class StagingArea(Entity):
    def __init__(self, area_id, latitude, longitude):
        super().__init__(entity_id=area_id, entity_type=self.__class__)
        self.location = Location(latitude, longitude)

    def reset(self):
        self.num_helicopters_serviced = 0

    def update_service_metrics(self):
        self.num_helicopters_serviced += 1

    def get_service_metrics(self):
        return {
            "num_helicopters_serviced": self.num_helicopters_serviced
        }

    def __repr__(self):
        return f"SA{self.id}"

    def to_dict(self):
        return {
            "id": self.id,
            "lat": self.location.latitude,
            "lon": self.location.longitude,
        }


class MilitaryTreatmentFacility(Entity):

    class Role(IntEnum):
        II = 2
        III = 3

    def __init__(
        self, 
        MTF_id, 
        role, 
        latitude, 
        longitude, 
        num_beds,
        ):
        super().__init__(entity_id=MTF_id, entity_type=self.__class__)
        self.role = MilitaryTreatmentFacility.Role(role)
        self.location = Location(latitude, longitude)
        self.num_beds = num_beds
        self.treatment_time_generators = {
            Request.Precedence.URGENT: RandomVariateGenerator(
            # distribution="triangular",
            # parameters={"left": 2, "mode": 3, "right": 5}, # hours
            distribution="truncnorm",
            parameters={
                "a": (12 - 48) / 12, # lower bound; (a - loc) / scale = (a - mean) / std
                "b": (72 - 48) / 12, # upper bound
                "loc": 48, # mean (hours)
                "scale": 12 # standard deviation (hours)
                },
            library="scipy"
            ), 
            Request.Precedence.PRIORITY: RandomVariateGenerator(
            # distribution="triangular",
            # parameters={"left": 0.5, "mode": 1, "right": 1.5}, # hours
            distribution="truncnorm",
            parameters={
                "a": (2 - 4) / 2, # lower bound; (a - loc) / scale = (a - mean) / std
                "b": (8 - 4) / 2, # upper bound
                "loc": 4, # mean (hours)
                "scale": 2 # standard deviation (hours)
                },
            library="scipy"
            ), 
            Request.Precedence.ROUTINE: RandomVariateGenerator(
            # distribution="triangular",
            # parameters={"left": 0.167, "mode": 0.5, "right": 0.75} # hours
            distribution="truncnorm",
            parameters={
                "a": (0.1 - 0.45) / 0.2, # lower bound; (a - loc) / scale = (a - mean) / std
                "b": (1 - 0.45) / 0.2, # upper bound
                "loc": 0.45, # mean (hours)
                "scale": 0.2 # standard deviation (hours)
                },
            library="scipy"
            ),
            }

    def __repr__(self):
        return f"MTF{self.id}"

    def reset(self):
        self.num_available_beds = self.num_beds
        self.treatment_times = np.zeros(self.num_beds)
        self.available_beds = list(range(self.num_beds))
        self.bed_reservations = defaultdict(list)
        new_rngs = self.environment.np_random.spawn(3)
        self.treatment_time_generators[Request.Precedence.URGENT].reset(new_rngs[0])
        self.treatment_time_generators[Request.Precedence.PRIORITY].reset(new_rngs[1])
        self.treatment_time_generators[Request.Precedence.ROUTINE].reset(new_rngs[2])

    def to_dict(self):
        return {
            "id": self.id,
            "lat": self.location.latitude,
            "lon": self.location.longitude,
            "role": self.role,
            "num_available_beds": self.num_available_beds,
            }

    def get_observation(self):
        return {
            "num_available_beds": self.num_available_beds / self.num_beds
        }

    def handle_event(self, event):
        if event.type == EventType.COMPLETED_PATIENT_TREATMENT:
            self.clear_bed(event.data["bed"])
            self.log(f"Completed treatment for patient in bed {event.data['bed']}")

    def reserve_beds(self, request):
        reserved_beds = [self.available_beds.pop() for _ in range(request.num_patients)]
        self.bed_reservations[request] = reserved_beds
        self.num_available_beds -= len(reserved_beds)
        self.log(f"Reserved {request.num_patients} beds for {request}")

    def get_num_reserved_beds(self, request):
        return len(self.bed_reservations[request])
        
    def cancel_bed_reservations(self, request):
        for bed in self.bed_reservations[request]:
            self.available_beds.append(bed)
        self.num_available_beds += len(self.bed_reservations[request])
        del self.bed_reservations[request]
        self.log(f"Canceled bed reservations for {request}")
        
    def receive_patients(self, request):
        precedences = [Request.Precedence.URGENT] * request.num_urgent + [Request.Precedence.PRIORITY] * request.num_priority + [Request.Precedence.ROUTINE] * request.num_routine
        for bed, precedence in zip(self.bed_reservations[request], precedences):
            treatment_time = self.treatment_time_generators[precedence].generate()
            self.treatment_times[bed] = treatment_time
            treatment_completion = Event(
                event_type=EventType.COMPLETED_PATIENT_TREATMENT, 
                handler=self, 
                time=self.environment.time + treatment_time, 
                data={"bed": bed}
                )
            self.schedule_event(treatment_completion)
        del self.bed_reservations[request]
        self.log(f"Received {request.num_patients} patients from {request}")

    def clear_bed(self, bed):
        self.treatment_times[bed] = 0
        self.available_beds.append(bed)
        self.num_available_beds += 1


class Helicopter(Entity):

    class Status(IntEnum):
        IDLE = 1
        FLIGHT_PREP = 2
        TO_PICKUP_SITE = 3
        LOADING = 4
        TO_MTF = 5
        OFFLOADING = 6
        TO_STAGING = 7
        REFUELING = 8

        @property
        def is_dispatchable(self):
            return self in {Helicopter.Status.IDLE, Helicopter.Status.TO_STAGING}

        @property
        def is_reroutable(self):
            return self in {Helicopter.Status.FLIGHT_PREP, Helicopter.Status.TO_PICKUP_SITE}

        def to_string(self):
            return {
                Helicopter.Status.IDLE: "idle at staging area",
                Helicopter.Status.FLIGHT_PREP: "conducting flight preparations",
                Helicopter.Status.TO_PICKUP_SITE: "en route to pickup_site",
                Helicopter.Status.LOADING: "loading patients at pickup_site",
                Helicopter.Status.TO_MTF: "en route to MTF",
                Helicopter.Status.OFFLOADING: "offloading patients at MTF",
                Helicopter.Status.TO_STAGING: "en route to staging area",
                Helicopter.Status.REFUELING: "refueling at staging area"
            }[self]

    def __init__(
        self, 
        helicopter_id, 
        initial_staging_area,
        avg_speed_knots=130, 
        fuel_endurance=2):
        super().__init__(entity_id=helicopter_id, entity_type=self.__class__)
        self.initial_staging_area = initial_staging_area
        self.speed = avg_speed_knots * 1852 # convert knots (nautical miles per hour) to meters per hour; 1 nm = 1852 m
        self.fuel_endurance = fuel_endurance # hours
        self.full_tank_range = self.speed * fuel_endurance # use meters to ensure compatibility with geographiclib
        self.flight_preparation_time_generator = RandomVariateGenerator(
            distribution="truncnorm",
            parameters={
                "a": (0.083 - 0.25) / 0.083, # lower bound; (a - loc) / scale = (a - mean) / std
                "b": (0.417 - 0.25) / 0.083, # upper bound
                "loc": 0.25, # mean
                "scale": 0.083 # hours
                },
            library="scipy"
            )
        self.loading_time_generator = RandomVariateGenerator(
            distribution="triangular",
            parameters={"left": 0.083, "mode": 0.167, "right": 0.25}
            )
        self.offloading_time_generator = RandomVariateGenerator(
            distribution="truncnorm",
            parameters={
                "a": (0.033 - 0.083) / 0.033, # lower bound; (a - loc) / scale = (a - mean) / std
                "b": (0.25 - 0.083) / 0.033, # upper bound
                "loc": 0.083, # mean
                "scale": 0.033 # hours
                },
            library="scipy"
            )
        self.refueling_time_generator = RandomVariateGenerator(
            distribution="truncnorm",
            parameters={
                "a": (0.083 - 0.25) / 0.083, # lower bound; (a - loc) / scale = (a - mean) / std
                "b": (0.417 - 0.25) / 0.083, # upper bound
                "loc": 0.25, # mean
                "scale": 0.083 # hours
                },
            library="scipy"
            )

    def __repr__(self):
        return f"H{self.id}"

    def reset(self):
        self.location = self.initial_staging_area.location.copy() # need to copy to avoid reference issues
        self.status = Helicopter.Status.IDLE
        self.distance_to_empty = self.full_tank_range
        self.request = None
        self.MTF = None
        self.staging_area = self.initial_staging_area
        self.next_event = None
        self.previous_event = None
        new_rngs = self.environment.np_random.spawn(4)
        self.flight_preparation_time_generator.reset(new_rngs[0])
        self.loading_time_generator.reset(new_rngs[1])
        self.offloading_time_generator.reset(new_rngs[2])
        self.refueling_time_generator.reset(new_rngs[3])
    
    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status.to_string(), 
            "lat": self.location.latitude,
            "lon": self.location.longitude,
            "distance_to_empty": self.distance_to_empty,
            "request": self.request.id if self.request else -1,
            "MTF": self.MTF.id if self.MTF else -1,
            "staging_area": self.staging_area.id if self.staging_area else -1
            }

    def get_observation(self):
        return {
            "status": [
                self.status == Helicopter.Status.IDLE,
                self.status == Helicopter.Status.FLIGHT_PREP,
                self.status == Helicopter.Status.TO_PICKUP_SITE,
                self.status == Helicopter.Status.LOADING,
                self.status == Helicopter.Status.TO_MTF,
                self.status == Helicopter.Status.OFFLOADING,
                self.status == Helicopter.Status.TO_STAGING,
                self.status == Helicopter.Status.REFUELING
            ],
            "location": {
                "latitude": utilities.normalize(self.location.latitude, self.environment.map["lat_min"], self.environment.map["lat_max"]),
                "longitude": utilities.normalize(self.location.longitude, self.environment.map["lon_min"], self.environment.map["lon_max"]),
            },
            "distance_to_empty": self.distance_to_empty / self.full_tank_range,
            "request": self.request.get_observation() if self.request else {
                "origin_zone": [False] * len(self.environment.zones),
                "location": {"latitude": 0, "longitude": 0},
                "num_patients": {Request.Precedence.URGENT: 0, Request.Precedence.PRIORITY: 0, Request.Precedence.ROUTINE: 0},
                "hours_waiting": 0
            },
            "mtf": [self.MTF == MTF if self.MTF else False for MTF in self.environment.MTFs],
            "staging_area": [self.staging_area == staging_area for staging_area in self.environment.staging_areas]
        }

    def update_state(self):
        if self.previous_event and self.previous_event.time < self.environment.time:
            match self.status:
                case Helicopter.Status.TO_PICKUP_SITE:
                    self.update_location(destination=self.request.pickup_site)
                case Helicopter.Status.TO_MTF:
                    self.update_location(destination=self.MTF.location)
                case Helicopter.Status.TO_STAGING:
                    self.update_location(destination=self.staging_area.location)

    def update_location(self, destination):
        time_traveled = self.environment.time - self.environment.previous_event_time
        distance_traveled = self.speed * time_traveled
        self.location.update(distance=distance_traveled, destination=destination)
        # Account for fuel consumption
        self.distance_to_empty -= distance_traveled
        self.log(f"Traveled {distance_traveled} meters to {destination} in {time_traveled} hours")

    def handle_event(self, event):
        match event.type:
            case EventType.RECEIVED_DECISION:
                if self.status.is_dispatchable:
                    self.log(f"Is {self.status.to_string()} and is initiating dispatch to {event.data['request']}, {event.data['MTF']}, {event.data['staging_area']}")
                    self.dispatch(event.data["request"], event.data["MTF"], event.data["staging_area"])
                    self.log(f"Dispatched to {self.request}, {self.MTF}, {self.staging_area}")
                elif self.status.is_reroutable:
                    self.log(f"Is {self.status.to_string()} and is initiating reroute to {event.data['request']}, {event.data['MTF']}, {event.data['staging_area']}")
                    self.reroute(event.data["request"], event.data["MTF"], event.data["staging_area"])
                    self.log(f"Rerouted to {self.request}, {self.MTF}, {self.staging_area}")
            case EventType.COMPLETED_FLIGHT_PREPARATIONS:
                self.log(f"Completed flight preparations at {self.staging_area}")
                self.depart_to_pickup_site()
                self.log(f"Departed to {self.request}'s pickup_site")
            case EventType.ARRIVED_AT_PICKUP_SITE:
                self.log(f"Arrived at {self.request}'s pickup_site")
                self.update_location(destination=self.request.pickup_site)
                self.begin_loading_patients()
                self.log(f"Began loading patients at {self.request}'s pickup_site")
            case EventType.LOADED_PATIENTS:
                self.log(f"Loaded patients at {self.request}'s pickup_site")
                self.depart_to_MTF()
                self.log(f"Departed to {self.MTF}")
            case EventType.ARRIVED_AT_MTF:
                self.log(f"Arrived at {self.MTF}")
                self.update_location(destination=self.MTF.location)
                self.begin_offloading_patients()
                self.log(f"Began offloading patients at {self.MTF}")
            case EventType.OFFLOADED_PATIENTS:
                self.log(f"Offloaded patients at {self.MTF}")
                self.environment.request_manager.complete_service(self, self.request, self.MTF)
                self.request = None
                self.MTF = None
                self.depart_to_staging_area()
                self.log(f"Departed to {self.staging_area}")
            case EventType.ARRIVED_AT_STAGING_AREA:
                self.log(f"Arrived at {self.staging_area}")
                self.update_location(destination=self.staging_area.location)
                self.begin_refueling()
                self.log(f"Began refueling at {self.staging_area}")
            case EventType.COMPLETED_REFUELING:
                fraction_refueled = (self.full_tank_range - self.distance_to_empty) / self.full_tank_range
                self.distance_to_empty = self.full_tank_range
                self.staging_area.update_service_metrics()
                self.log(f"Refueled {fraction_refueled:.2f} tank at {self.staging_area}")
                self.idle()

        self.previous_event = event

    def receive_decision(self, request, MTF, staging_area):
        self.request = request
        self.MTF = MTF
        self.staging_area = staging_area
        self.environment.request_manager.begin_service(self, request, MTF)
        
    def dispatch(self, request, MTF, staging_area):
        self.receive_decision(request, MTF, staging_area)
        if self.status == Helicopter.Status.TO_STAGING:
            self.cancel_event(self.next_event)
            self.depart_to_pickup_site()
        else:
            self.conduct_flight_preparations()

    def reroute(self, request, MTF, staging_area):
        self.environment.request_manager.cancel_service(self, self.request, self.MTF)
        self.receive_decision(request, MTF, staging_area)
        if self.status == Helicopter.Status.TO_PICKUP_SITE:
            self.cancel_event(self.next_event)
            self.depart_to_pickup_site()
        self.log(f"{self}'s status is {self.status.to_string()} and next event is {self.next_event}")

    def conduct_flight_preparations(self):
        preparation_time = self.flight_preparation_time_generator.generate()
        completion = Event(
            event_type=EventType.COMPLETED_FLIGHT_PREPARATIONS, 
            handler=self, 
            time=self.environment.time + preparation_time
            )
        self.schedule_event(completion)
        self.next_event = completion
        self.status = Helicopter.Status.FLIGHT_PREP

    def depart_to_pickup_site(self):
        distance = self.location.distance_to(self.request.pickup_site)
        travel_time = distance / self.speed
        arrival = Event(event_type=EventType.ARRIVED_AT_PICKUP_SITE, handler=self, time=self.environment.time + travel_time)
        self.schedule_event(arrival)
        self.next_event = arrival
        self.status = Helicopter.Status.TO_PICKUP_SITE

    def begin_loading_patients(self):
        loading_time = self.loading_time_generator.generate()
        completion = Event(event_type=EventType.LOADED_PATIENTS, handler=self, time=self.environment.time + loading_time)
        self.schedule_event(completion)
        self.next_event = completion
        self.status = Helicopter.Status.LOADING

    def depart_to_MTF(self):
        distance = self.location.distance_to(self.MTF.location)
        travel_time = distance / self.speed
        arrival = Event(event_type=EventType.ARRIVED_AT_MTF, handler=self, time=self.environment.time + travel_time)
        self.schedule_event(arrival)
        self.next_event = arrival
        self.status = Helicopter.Status.TO_MTF

    def begin_offloading_patients(self):
        offloading_time = self.offloading_time_generator.generate()
        completion = Event(event_type=EventType.OFFLOADED_PATIENTS, handler=self, time=self.environment.time + offloading_time)
        self.schedule_event(completion)
        self.next_event = completion
        self.status = Helicopter.Status.OFFLOADING

    def depart_to_staging_area(self):
        distance = self.location.distance_to(self.staging_area.location)
        travel_time = distance / self.speed
        arrival = Event(event_type=EventType.ARRIVED_AT_STAGING_AREA, handler=self, time=self.environment.time + travel_time)
        self.schedule_event(arrival)
        self.next_event = arrival
        self.status = Helicopter.Status.TO_STAGING

    def begin_refueling(self):
        refueling_time = self.refueling_time_generator.generate()
        completion = Event(event_type=EventType.COMPLETED_REFUELING, handler=self, time=self.environment.time + refueling_time)
        self.environment.schedule_event(completion)
        self.next_event = completion
        self.status = Helicopter.Status.REFUELING

    def idle(self):
        self.next_event = None
        self.status = Helicopter.Status.IDLE


class Request(Entity):

    class Precedence(Enum): # value is 1 / hour_limit        
        URGENT = 1
        PRIORITY = 1/4
        ROUTINE = 1/24

        @property
        def importance_weight(self):
            return {
                Request.Precedence.URGENT: 1, 
                Request.Precedence.PRIORITY: 1/10, 
                Request.Precedence.ROUTINE: 1/100,
                }[self]

        @property
        def hour_limit(self):
            return {
                Request.Precedence.URGENT: 1, 
                Request.Precedence.PRIORITY: 4, 
                Request.Precedence.ROUTINE: 24
                }[self]

        def to_string(self):
            return {
                Request.Precedence.URGENT: "Urgent",
                Request.Precedence.PRIORITY: "Priority",
                Request.Precedence.ROUTINE: "Routine"
            }[self]

        def __lt__(self, other):
            return self.value < other.value

        def __repr__(self):
            return self.to_string()

    def __init__(self, request_id, arrival_time, origin_zone, pickup_site, num_urgent, num_priority, num_routine):
        super().__init__(entity_id=request_id, entity_type=self.__class__)
        self.arrival_time = arrival_time
        self.origin_zone = origin_zone
        self.pickup_site = pickup_site
        self.num_urgent = num_urgent
        self.num_priority = num_priority
        self.num_routine = num_routine
        self.num_patients = num_urgent + num_priority + num_routine
        self.precedence = Request.Precedence.URGENT if num_urgent > 0 else Request.Precedence.PRIORITY if num_priority > 0 else Request.Precedence.ROUTINE
        self.required_MTF_role = MilitaryTreatmentFacility.Role.III if num_urgent > 0 else MilitaryTreatmentFacility.Role.II
        self.deadline = arrival_time + self.precedence.hour_limit
        self.hours_past_deadline = 0
        self.hours_waiting = 0
        self.service_completion_time = None
        self.assigned_helicopter = None
        self.assigned_MTF = None

    def __repr__(self):
        return f"R{self.id}"

    def __lt__(self, other):
        if self.hours_past_deadline == other.hours_past_deadline:
            if self.precedence == other.precedence:
                return self.hours_waiting > other.hours_waiting
            return self.precedence > other.precedence
        return self.hours_past_deadline > other.hours_past_deadline

    def initialize(self, environment):
        super().initialize(environment)
        if self.origin_zone is not None:
            self.origin_zone.active_requests[self.precedence].add(self)
            if self.environment.forecast_recency_bias:
                self.origin_zone.update_belief_state(self)

    def to_dict(self):
        return {
            "id": self.id,
            "arrival_time": self.arrival_time,
            "origin_zone": self.origin_zone.id,
            "pickup_site_lat": self.pickup_site.latitude,
            "pickup_site_lon": self.pickup_site.longitude,
            "num_urgent": self.num_urgent,
            "num_priority": self.num_priority,
            "num_routine": self.num_routine,
            "precedence": self.precedence.to_string(),
            "required_MTF_role": self.required_MTF_role,
            "deadline": self.deadline,
            "hours_past_deadline": self.hours_past_deadline,
            "hours_waiting": self.hours_waiting,
            "service_completion_time": self.service_completion_time,
            "assigned_helicopter": self.assigned_helicopter.id if self.assigned_helicopter else None,
            "assigned_MTF": self.assigned_MTF.id if self.assigned_MTF else None
            }

    def get_observation(self):
        return {
            "origin_zone": [self.origin_zone is not None and self.origin_zone.id == zone.id for zone in self.environment.zones],
            "location": {
                "latitude": utilities.normalize(self.pickup_site.latitude, self.environment.map["lat_min"], self.environment.map["lat_max"]),
                "longitude": utilities.normalize(self.pickup_site.longitude, self.environment.map["lon_min"], self.environment.map["lon_max"]),
            },
            "num_patients": {
                Request.Precedence.URGENT: self.num_urgent / 4, # For the purposes of simulation, there are never more than 4 patients per request
                Request.Precedence.PRIORITY: self.num_priority / 4,
                Request.Precedence.ROUTINE: self.num_routine / 4
            },
            "hours_waiting": min(self.hours_waiting, 12) / 12, # clip at 12 hours
        }

    def update_state(self):
        self.hours_waiting = self.environment.time - self.arrival_time
        self.hours_past_deadline = max(0, self.environment.time - self.deadline)


class RequestManager(Entity):

    def __init__(self):
        super().__init__(entity_id="RM", entity_type=self.__class__)

    def __repr__(self):
        return f"RM"

    def reset(self):
        self.requests_to_examine = []
        self.requests_to_revisit = []
        self.requests_being_serviced = {helicopter: None for helicopter in self.environment.helicopters}

        self.avg_service_times = {mtf: {precedence: 0 for precedence in Request.Precedence} for mtf in self.environment.MTFs}
        self.total_num_requests_serviced = {mtf: {precedence: 0 for precedence in Request.Precedence} for mtf in self.environment.MTFs}
        self.num_requests_serviced_on_time = {mtf: {precedence: 0 for precedence in Request.Precedence} for mtf in self.environment.MTFs}
        self.percent_requests_serviced_on_time = {mtf: {precedence: 0 for precedence in Request.Precedence} for mtf in self.environment.MTFs}

    def update_service_metrics(self, request):
        if request.assigned_MTF is None:
            print(f"{request}, {request.precedence}, has no assigned MTF")
            self.environment.to_table()
            print(f"Env's future events: {self.environment.future_events.peek()}")
            print(f"Env's previous event: {self.environment.previous_event}")
            print(f"{request}'s assigned helicopter: {request.assigned_helicopter}")
            return
        self.total_num_requests_serviced[request.assigned_MTF][request.precedence] += 1
        if request.hours_waiting <= request.precedence.hour_limit:
            self.num_requests_serviced_on_time[request.assigned_MTF][request.precedence] += 1
        self.avg_service_times[request.assigned_MTF][request.precedence] = (
            self.avg_service_times[request.assigned_MTF][request.precedence] * 
            (self.total_num_requests_serviced[request.assigned_MTF][request.precedence] - 1) + 
            request.hours_waiting
        ) / self.total_num_requests_serviced[request.assigned_MTF][request.precedence]
        self.percent_requests_serviced_on_time[request.assigned_MTF][request.precedence] = (
            self.num_requests_serviced_on_time[request.assigned_MTF][request.precedence] / 
            self.total_num_requests_serviced[request.assigned_MTF][request.precedence]
        ) * 100

    def get_service_metrics(self):
        return {
            "avg_service_times": self.avg_service_times,
            "total_num_requests_serviced": self.total_num_requests_serviced,
            "num_requests_serviced_on_time": self.num_requests_serviced_on_time,
            "percent_requests_serviced_on_time": self.percent_requests_serviced_on_time,
        }

    def to_dict(self):
        return {
            "requests_to_examine": [request.to_dict() for request in self.requests_to_examine],
            "requests_to_revisit": [request.to_dict() for request in self.requests_to_revisit],
            "requests_being_serviced": [request.to_dict() for request in self.requests_being_serviced],
            }

    def update_state(self):
        for request in filter(None, itertools.chain(self.requests_to_examine, self.requests_to_revisit, self.requests_being_serviced.values())):
            request.update_state()

    @property
    def has_unexamined_requests(self):
        return bool(self.requests_to_examine)

    def handle_event(self, event):
        super().handle_event(event)
        if event.type == EventType.RECEIVED_REQUEST:
            self.add_request_to_examine(event.data["request"])

    def add_request_to_examine(self, request):
        if request.arrival_time == self.environment.time:
            request.initialize(self.environment)
        self.requests_to_examine.append(request)
        self.requests_to_examine.sort()
        self.log(f"{request} added to requests to examine: {self.requests_to_examine}")

    def get_next_request_to_examine(self):
        if self.requests_to_examine:
            request = self.requests_to_examine.pop(0)
            self.log(f"{request} popped from requests to examine")
            return request
        return None

    def add_request_to_revisit(self, request):
        self.requests_to_revisit.append(request)
        self.log(f"Request {request} added to requests to revisit {self.requests_to_revisit}")

    def reset_unexamined_requests(self):
        if self.requests_to_revisit:
            self.requests_to_examine.extend(self.requests_to_revisit)
            self.requests_to_examine.sort()
            self.requests_to_revisit = []
            self.log(f"Requests to examine reset: {self.requests_to_examine}")

    def begin_service(self, helicopter, request, MTF):
        MTF.reserve_beds(request)
        self.requests_being_serviced[helicopter] = request
        request.assigned_helicopter = helicopter
        self.log(f"{request} is assigned to {request.assigned_helicopter}")
        request.assigned_MTF = MTF
        self.log(f"{request} is assigned to {request.assigned_MTF}")
        self.log(f"{request} began service by {helicopter}")

    def complete_service(self, helicopter, request, MTF):
        MTF.receive_patients(request)
        self.requests_being_serviced[helicopter] = None
        self.update_service_metrics(request)
        request.origin_zone.active_requests[request.precedence].remove(request)
        request.origin_zone.update_service_metrics(request)

        x = self.environment.time - request.arrival_time
        for _ in range(request.num_urgent):
            self.environment.reward += 1 / (self.environment.time - request.arrival_time)
        for _ in range(request.num_priority):
            self.environment.reward += 1 / (4 * (self.environment.time - request.arrival_time))
        for _ in range(request.num_routine):
            self.environment.reward += 1 / (24 * (self.environment.time - request.arrival_time))

        request.service_completion_time = self.environment.time
        self.log(f"{request} completed service by {helicopter}.")

    def cancel_service(self, helicopter, request, MTF):
        MTF.cancel_bed_reservations(request)
        self.requests_being_serviced[helicopter] = None
        self.add_request_to_examine(request)
        request.assigned_helicopter = None
        request.assigned_MTF = None
        self.log(f"{request} removed from requests being serviced")


class Zone(Entity):

    NORMALIZATION_UPPER_BOUNDS = {
        "num_active_requests": 10,
        "num_patients": 40,
        "cumulative_total_wait_time": 480, # 40 patients * 12 hours
        "mean_arrival_rate": 10,
    }

    def __init__(self, zone_id, lat_min, lat_max, lon_min, lon_max, alpha):
        super().__init__(entity_id=zone_id, entity_type=self.__class__)
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.alpha = alpha
        
    def __repr__(self):
        return f"Z{self.id}"

    def reset(self):
        self.active_requests = {
            Request.Precedence.URGENT: set(),
            Request.Precedence.PRIORITY: set(),
            Request.Precedence.ROUTINE: set()
        }
        self.request_arrival_counts = {precedence: 0 for precedence in Request.Precedence}
        self.previous_request_arrival_times = {precedence: 0 for precedence in Request.Precedence}
        self.estimated_mean_arrival_rates = {precedence: 0 for precedence in Request.Precedence}

        self.avg_service_times = {mtf: {precedence: 0 for precedence in Request.Precedence} for mtf in self.environment.MTFs}
        self.total_num_requests_serviced = {mtf: {precedence: 0 for precedence in Request.Precedence} for mtf in self.environment.MTFs}
        self.num_requests_serviced_on_time = {mtf: {precedence: 0 for precedence in Request.Precedence} for mtf in self.environment.MTFs}
        self.percent_requests_serviced_on_time = {mtf: {precedence: 0 for precedence in Request.Precedence} for mtf in self.environment.MTFs}

    def update_service_metrics(self, request):
        if request.assigned_MTF is None:
            print(f"Env time: {self.environment.time}")
            print(f"{request}, {request.precedence}, has no assigned MTF")
            self.environment.to_table()
            print(f"Env's future events: {self.environment.future_events.peek()}")
            print(f"Env's previous event: {self.environment.previous_event}")
            print(f"{request}'s assigned helicopter: {request.assigned_helicopter}")
            for helicopter in self.environment.helicopters:
                print(f"{helicopter}'s previous event: {helicopter.previous_event}")
                print(f"{helicopter}'s next event: {helicopter.next_event}")
                print(f"{helicopter}'s status: {helicopter.status}")
                print(f"{helicopter}'s request {helicopter.request}")
                print(f"{helicopter}'s MTF {helicopter.MTF}")
                print(f"{helicopter}'s staging area {helicopter.staging_area}")
            return
        self.total_num_requests_serviced[request.assigned_MTF][request.precedence] += 1
        if request.hours_waiting <= request.precedence.hour_limit:
            self.num_requests_serviced_on_time[request.assigned_MTF][request.precedence] += 1
        self.avg_service_times[request.assigned_MTF][request.precedence] = (
            self.avg_service_times[request.assigned_MTF][request.precedence] * 
            (self.total_num_requests_serviced[request.assigned_MTF][request.precedence] - 1) + 
            request.hours_waiting
        ) / self.total_num_requests_serviced[request.assigned_MTF][request.precedence]
        self.percent_requests_serviced_on_time[request.assigned_MTF][request.precedence] = (
            self.num_requests_serviced_on_time[request.assigned_MTF][request.precedence] / 
            self.total_num_requests_serviced[request.assigned_MTF][request.precedence]
        ) * 100

    def get_service_metrics(self):
        return {
            "avg_service_times": self.avg_service_times,
            "total_num_requests_serviced": self.total_num_requests_serviced,
            "num_requests_serviced_on_time": self.num_requests_serviced_on_time,
            "percent_requests_serviced_on_time": self.percent_requests_serviced_on_time,
        }

    def to_dict(self):
        return {
            "id": self.id,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "urgent_rate": self.estimated_mean_arrival_rates[Request.Precedence.URGENT],
            "priority_rate": self.estimated_mean_arrival_rates[Request.Precedence.PRIORITY],
            "routine_rate": self.estimated_mean_arrival_rates[Request.Precedence.ROUTINE],
        }

    def get_observation(self):
        # clip and normalize values
        if self.alpha:
            return {
                "information_state": {
                    "num_active_requests": {
                        Request.Precedence.URGENT: min(
                                len(self.active_requests[Request.Precedence.URGENT]), 
                                self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"],
                        Request.Precedence.PRIORITY: min(
                                len(self.active_requests[Request.Precedence.PRIORITY]), 
                                self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"],
                        Request.Precedence.ROUTINE: min(
                                len(self.active_requests[Request.Precedence.ROUTINE]), 
                                self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"],
                    },
                    "num_patients": {
                        Request.Precedence.URGENT: min(
                                sum(request.num_urgent for request in self.active_requests[Request.Precedence.URGENT]), 
                                self.NORMALIZATION_UPPER_BOUNDS["num_patients"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_patients"],
                        Request.Precedence.PRIORITY: min(
                                sum(
                                    request.num_priority 
                                    for request in itertools.chain(
                                        self.active_requests[Request.Precedence.URGENT], 
                                        self.active_requests[Request.Precedence.PRIORITY]
                                    )
                                ), 
                                self.NORMALIZATION_UPPER_BOUNDS["num_patients"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_patients"],
                        Request.Precedence.ROUTINE: min(
                                sum(
                                    request.num_routine 
                                    for request in itertools.chain(
                                        self.active_requests[Request.Precedence.URGENT],
                                        self.active_requests[Request.Precedence.PRIORITY],
                                        self.active_requests[Request.Precedence.ROUTINE]
                                    )
                                ),
                                self.NORMALIZATION_UPPER_BOUNDS["num_patients"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_patients"],
                    },
                    "cumulative_total_wait_time": {
                        Request.Precedence.URGENT: min(
                                sum(request.num_urgent * request.hours_waiting for request in self.active_requests[Request.Precedence.URGENT]), 
                                self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"],
                        Request.Precedence.PRIORITY: min(
                                sum(
                                    request.num_priority * request.hours_waiting
                                    for request in itertools.chain(
                                        self.active_requests[Request.Precedence.URGENT], 
                                        self.active_requests[Request.Precedence.PRIORITY]
                                    )
                                ), 
                                self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"],
                        Request.Precedence.ROUTINE: min(
                                sum(
                                    request.num_routine * request.hours_waiting
                                    for request in itertools.chain(
                                        self.active_requests[Request.Precedence.URGENT],
                                        self.active_requests[Request.Precedence.PRIORITY],
                                        self.active_requests[Request.Precedence.ROUTINE]
                                    )
                                ),
                                self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"],
                    },
                },
                "belief_state": {
                    "mean_arrival_rates": {
                        Request.Precedence.URGENT: min(
                            self.estimated_mean_arrival_rates[Request.Precedence.URGENT], 
                            self.NORMALIZATION_UPPER_BOUNDS["mean_arrival_rate"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["mean_arrival_rate"],
                        Request.Precedence.PRIORITY: min(
                            self.estimated_mean_arrival_rates[Request.Precedence.PRIORITY], 
                            self.NORMALIZATION_UPPER_BOUNDS["mean_arrival_rate"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["mean_arrival_rate"],
                        Request.Precedence.ROUTINE: min(
                            self.estimated_mean_arrival_rates[Request.Precedence.ROUTINE], 
                            self.NORMALIZATION_UPPER_BOUNDS["mean_arrival_rate"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["mean_arrival_rate"],
                    },
                }
            }
        
        else:

            return {
                "information_state": {
                    "num_active_requests": {
                        Request.Precedence.URGENT: min(
                                len(self.active_requests[Request.Precedence.URGENT]), 
                                self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"],
                        Request.Precedence.PRIORITY: min(
                                len(self.active_requests[Request.Precedence.PRIORITY]), 
                                self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"],
                        Request.Precedence.ROUTINE: min(
                                len(self.active_requests[Request.Precedence.ROUTINE]), 
                                self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_active_requests"],
                    },
                    "num_patients": {
                        Request.Precedence.URGENT: min(
                                sum(request.num_urgent for request in self.active_requests[Request.Precedence.URGENT]), 
                                self.NORMALIZATION_UPPER_BOUNDS["num_patients"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_patients"],
                        Request.Precedence.PRIORITY: min(
                                sum(
                                    request.num_priority 
                                    for request in itertools.chain(
                                        self.active_requests[Request.Precedence.URGENT], 
                                        self.active_requests[Request.Precedence.PRIORITY]
                                    )
                                ), 
                                self.NORMALIZATION_UPPER_BOUNDS["num_patients"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_patients"],
                        Request.Precedence.ROUTINE: min(
                                sum(
                                    request.num_routine 
                                    for request in itertools.chain(
                                        self.active_requests[Request.Precedence.URGENT],
                                        self.active_requests[Request.Precedence.PRIORITY],
                                        self.active_requests[Request.Precedence.ROUTINE]
                                    )
                                ),
                                self.NORMALIZATION_UPPER_BOUNDS["num_patients"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["num_patients"],
                    },
                    "cumulative_total_wait_time": {
                        Request.Precedence.URGENT: min(
                                sum(request.num_urgent * request.hours_waiting for request in self.active_requests[Request.Precedence.URGENT]), 
                                self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"],
                        Request.Precedence.PRIORITY: min(
                                sum(
                                    request.num_priority * request.hours_waiting
                                    for request in itertools.chain(
                                        self.active_requests[Request.Precedence.URGENT], 
                                        self.active_requests[Request.Precedence.PRIORITY]
                                    )
                                ), 
                                self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"],
                        Request.Precedence.ROUTINE: min(
                                sum(
                                    request.num_routine * request.hours_waiting
                                    for request in itertools.chain(
                                        self.active_requests[Request.Precedence.URGENT],
                                        self.active_requests[Request.Precedence.PRIORITY],
                                        self.active_requests[Request.Precedence.ROUTINE]
                                    )
                                ),
                                self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"]
                            ) / self.NORMALIZATION_UPPER_BOUNDS["cumulative_total_wait_time"],
                    },
                },
            }


    def update_belief_state(self, request):
        observed_interarrival_time = request.arrival_time - self.previous_request_arrival_times[request.precedence]
        self.log(f"Observed interarrival time for {request.precedence} requests: {observed_interarrival_time}")
        if observed_interarrival_time > 0:
            observed_arrival_rate = 1 / observed_interarrival_time
        else:
            observed_arrival_rate = 0

        self.previous_request_arrival_times[request.precedence] = request.arrival_time
        self.request_arrival_counts[request.precedence] += 1

        if self.request_arrival_counts[request.precedence] == 1:
            self.estimated_mean_arrival_rates[request.precedence] = observed_arrival_rate
        else:
            # higher alpha means more weight is given to the most recent observation
            self.estimated_mean_arrival_rates[request.precedence] = self.alpha * observed_arrival_rate + (1 - self.alpha) * self.estimated_mean_arrival_rates[request.precedence]
            self.log(f"Estimated arrival rate mean for {request.precedence} requests: {self.estimated_mean_arrival_rates[request.precedence]}")
            
    def contains(self, location):
        # return self.lat_min <= location.latitude <= self.lat_max and self.lon_min <= location.longitude <= self.lon_max
        epsilon = 1e-6
        return (self.lat_min - epsilon <= location.latitude <= self.lat_max + epsilon and
                self.lon_min - epsilon <= location.longitude <= self.lon_max + epsilon)

class MedevacDispatchingEnvironment(DiscreteEventEnvironment):
    def __init__(
        self,  
        map_config_file,
        MTF_config_file, 
        staging_area_config_file, 
        casualty_cluster_center_config_file,
        intensity_function_config_file,
        num_zones,
        forecast_recency_bias,
        time_limit=24*7, # needs to be 24 * 7 because of domain of intensity functions
        excitation_factor=0.4,
        decay_rate=1, 
        verbose=False
        ):

        super().__init__(time_limit, verbose)

        self.map_config_file = map_config_file
        self.MTF_config_file = MTF_config_file
        self.staging_area_config_file = staging_area_config_file
        self.casualty_cluster_center_config_file = casualty_cluster_center_config_file
        self.intensity_function_config_file = intensity_function_config_file
        self.num_zones = num_zones
        self.forecast_recency_bias = forecast_recency_bias
        self.excitation_factor = excitation_factor
        self.decay_rate = decay_rate

        self.map = {}
        for lat_min, lat_max, lon_min, lon_max in (
            line.strip().split(",") for line in open(map_config_file).readlines()[1:]
        ):
            self.map = {
                "lat_min": float(lat_min),
                "lat_max": float(lat_max),
                "lon_min": float(lon_min),
                "lon_max": float(lon_max)
            }

        self.generate_zones(num_zones)

        self.casualty_cluster_centers = [
            Location(float(latitude), float(longitude))
            for latitude, longitude in (
                line.strip().split(",") for line in open(casualty_cluster_center_config_file).readlines()[1:]
            )
        ]
        
        self.MTF_id_generator = utilities.IdGenerator()
        self.MTFs = [
            MilitaryTreatmentFacility(self.MTF_id_generator.next(), int(role), float(latitude), float(longitude), int(num_beds))
            for role, latitude, longitude, num_beds in (
                line.strip().split(",") for line in open(MTF_config_file).readlines()[1:]
            )
        ]

        self.staging_area_id_generator = utilities.IdGenerator()
        self.helicopter_id_generator = utilities.IdGenerator()
        self.staging_areas = []
        self.helicopters = []
        for latitude, longitude, num_helicopters in (
            line.strip().split(",") for line in open(staging_area_config_file).readlines()[1:]
        ):
            staging_area = StagingArea(self.staging_area_id_generator.next(), float(latitude), float(longitude))
            self.staging_areas.append(staging_area)
            self.helicopters.extend(
            Helicopter(self.helicopter_id_generator.next(), staging_area) for _ in range(int(num_helicopters))
            )

        self.intensity_functions = []
        self.max_intensities = []
        with open(intensity_function_config_file, "r") as file:
            reader = csv.reader(file)
            header = next(reader) # first row contains the domain values
            self.domain = np.array(header[1:], dtype=float)
            
            # Read the rows (skip the first column which contains labels like 'Function_1')
            for row in reader:
                intensity_function_range = np.array(row[1:], dtype=float)
                max_intensity = np.max(intensity_function_range)
                intensity_function = scipy.interpolate.CubicSpline(self.domain, intensity_function_range)

                self.intensity_functions.append(intensity_function)
                self.max_intensities.append(max_intensity)

        self.decision_space = list(itertools.product(self.helicopters, self.MTFs, self.staging_areas))
        self.decision_space.append((None, None, None)) # Null action, i.e., revisit request (no assignment)
        self.decision_index_map = {index: decision for index, decision in enumerate(self.decision_space)}

        # if the max queue size is too small, requests will be lost from the system
        self.request_manager = RequestManager()

        for entity in itertools.chain(self.zones, self.MTFs, self.helicopters, [self.request_manager], self.staging_areas):
            self.add_entity(entity)

        self.action_space = gym.spaces.Discrete(len(self.decision_space))

        self.request_observation_space = gym.spaces.Dict(
            {
                "origin_zone": gym.spaces.MultiBinary(len(self.zones)),
                "location": gym.spaces.Dict(
                    {
                        "latitude": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), 
                        "longitude": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
                    }
                ),
                "num_patients": gym.spaces.Dict(
                    {
                        Request.Precedence.URGENT: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                        Request.Precedence.PRIORITY: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                        Request.Precedence.ROUTINE: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    }
                ),
                "hours_waiting": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )

        self.helicopter_observation_space = gym.spaces.Dict(
            {
                "status": gym.spaces.MultiBinary(8),
                "location": gym.spaces.Dict(
                    {
                    "latitude": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), 
                    "longitude": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
                    }
                ),
                "distance_to_empty": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "request": self.request_observation_space,
                "mtf": gym.spaces.MultiBinary(len(self.MTFs)),
                "staging_area": gym.spaces.MultiBinary(len(self.staging_areas)),
            }
        )

        self.MTF_observation_space = gym.spaces.Dict(
            {
                "num_available_beds": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )

        if self.forecast_recency_bias:

            self.zone_observation_space = gym.spaces.Dict(
                {
                    "information_state": gym.spaces.Dict(
                        {
                            "num_active_requests": gym.spaces.Dict(
                                {
                                    Request.Precedence.URGENT: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.PRIORITY: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.ROUTINE: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                }
                            ),
                            "num_patients": gym.spaces.Dict(
                                {
                                    Request.Precedence.URGENT: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.PRIORITY: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.ROUTINE: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                }
                            ),
                            "cumulative_total_wait_time": gym.spaces.Dict(
                                {
                                    Request.Precedence.URGENT: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.PRIORITY: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.ROUTINE: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                }
                            ),
                        }
                    ),
                    "belief_state": gym.spaces.Dict(
                        {
                            "mean_arrival_rates": gym.spaces.Dict(
                                {
                                    Request.Precedence.URGENT: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.PRIORITY: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.ROUTINE: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                }
                            ),
                        }
                    )
                }
            )

        else:

                self.zone_observation_space = gym.spaces.Dict(
                {
                    "information_state": gym.spaces.Dict(
                        {
                            "num_active_requests": gym.spaces.Dict(
                                {
                                    Request.Precedence.URGENT: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.PRIORITY: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.ROUTINE: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                }
                            ),
                            "num_patients": gym.spaces.Dict(
                                {
                                    Request.Precedence.URGENT: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.PRIORITY: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.ROUTINE: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                }
                            ),
                            "cumulative_total_wait_time": gym.spaces.Dict(
                                {
                                    Request.Precedence.URGENT: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.PRIORITY: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                    Request.Precedence.ROUTINE: gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                }
                            ),
                        }
                    ),
                }
            )

        self.observation_space = gym.spaces.Dict(
            {
                "time": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "request_under_consideration": self.request_observation_space,
                "mission_distances": gym.spaces.Box(low=0, high=1, shape=(len(self.decision_space),), dtype=np.float32),
                "helicopters": gym.spaces.Dict({helicopter.id: self.helicopter_observation_space for helicopter in self.helicopters}),
                "mtfs": gym.spaces.Dict({mtf.id: self.MTF_observation_space for mtf in self.MTFs}),
                "zones": gym.spaces.Dict({zone.id: self.zone_observation_space for zone in self.zones})
            }
        )
        self.parent_interarrival_time_generator = RandomVariateGenerator(
            distribution="exponential",
            parameters=None # set parameters in reset(); scale is dependent on Gaussian process sample
        )
        self.child_interarrival_time_generator = RandomVariateGenerator(
            distribution="exponential",
            parameters={"scale": self.decay_rate} # scale = decay_rate, i.e., beta
        )
        self.child_count_generator = RandomVariateGenerator(
            distribution="poisson",
            parameters={"lam": self.excitation_factor / self.decay_rate} # lambda = branching_ratio = excitation_factor / decay_rate
        )
        self.acceptance_probability_generator = RandomVariateGenerator(
            distribution="random",
            parameters={}
        )
        self.casualty_cluster_center_generator = RandomVariateGenerator(
            distribution="choice",
            parameters={"a": self.casualty_cluster_centers}
        )
        self.num_patients_generator = RandomVariateGenerator(
            distribution="choice",
            parameters={
                "a": [1, 2, 3, 4],
                "p": [0.574, 0.36, 0.05, 0.016]
                }
        )
        self.precedence_probability_generator = RandomVariateGenerator(
            distribution="random",
            parameters={}
        )
        self.intensity_function_index_generator = RandomVariateGenerator(
            distribution="integers",
            parameters={"low": 0, "high": len(self.intensity_functions)}
        )

    def copy(self):
        return MedevacDispatchingEnvironment(
            map_config_file=self.map_config_file,
            MTF_config_file=self.MTF_config_file,
            staging_area_config_file=self.staging_area_config_file,
            casualty_cluster_center_config_file=self.casualty_cluster_center_config_file,
            intensity_function_config_file=self.intensity_function_config_file,
            time_limit=self.time_limit,
            num_zones=self.num_zones,
            forecast_recency_bias=self.forecast_recency_bias,
            excitation_factor=self.excitation_factor,
            decay_rate=self.decay_rate,
            verbose=self.verbose
        )

    def reset(self, seed = None, options = None):
        super().reset(seed)

        new_rngs = self.np_random.spawn(9)
        self.parent_interarrival_time_generator.reset(new_rngs[0])
        self.child_interarrival_time_generator.reset(new_rngs[1])
        self.child_count_generator.reset(new_rngs[2])
        self.acceptance_probability_generator.reset(new_rngs[3])
        self.casualty_cluster_center_generator.reset(new_rngs[4])
        self.pickup_site_rng = new_rngs[5]
        self.num_patients_generator.reset(new_rngs[6])
        self.precedence_probability_generator.reset(new_rngs[7])
        self.intensity_function_index_generator.reset(new_rngs[8])

        self.request_id_generator = utilities.IdGenerator()
        self.generate_future_requests()

        _ = self.process_next_event()

        self.request_to_assign = self.request_manager.get_next_request_to_examine()

        feasible_decision_info = self.get_feasible_decision_info(self.request_to_assign)

        self.num_steps = 0

        return self.get_observation(feasible_decision_info), feasible_decision_info


    def to_dict(self):
        return {
            "time": self.time,
            "requests": [request.to_dict() if request else None for request in self.request_manager.requests],
            "MTFs": [MTF.to_dict() for MTF in self.MTFs],
            "staging_areas": [staging_area.to_dict() for staging_area in self.staging_areas],
            "helicopters": [helicopter.to_dict() for helicopter in self.helicopters],
            "zones": [zone.to_dict() for zone in self.zones],
            "feasible_decision_info": self.get_feasible_decision_info(self.request_manager.requests_to_examine[0])
        }

    def get_observation(self, feasible_decision_info):
        mission_distances = np.array([481520] * len(self.decision_space))
        if feasible_decision_info is not None:
            for index, distance in feasible_decision_info["mission_distances"]:
                mission_distances[index] = min(distance, 481520) # clip at 481520 m -> 130 nm/hr (avg speed) * 1852 m/nm * 2 hrs (fuel endurance)
        return {
            "time": self.time / self.time_limit,
            "request_under_consideration": self.request_to_assign.get_observation() if self.request_to_assign else {
                "origin_zone": [False] * len(self.zones),
                "location": {"latitude": 0, "longitude": 0},
                "num_patients": {Request.Precedence.URGENT: 0, Request.Precedence.PRIORITY: 0, Request.Precedence.ROUTINE: 0},
                "hours_waiting": 0
            },
            "mission_distances": mission_distances / 481520,
            "helicopters": {helicopter.id: helicopter.get_observation() for helicopter in self.helicopters},
            "mtfs": {mtf.id: mtf.get_observation() for mtf in self.MTFs},
            "zones": {zone.id: zone.get_observation() for zone in self.zones}
        }

    def generate_zones(self, num_zones):
            # Calculate the grid dimensions
            lat_min = self.map["lat_min"]
            lat_max = self.map["lat_max"]
            lon_min = self.map["lon_min"]
            lon_max = self.map["lon_max"]
            aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
            num_rows = int(math.sqrt(num_zones * aspect_ratio))
            num_cols = int(math.ceil(num_zones / num_rows))

            # Adjust if the rows and columns don't multiply exactly to num_zones
            if num_rows * num_cols < num_zones:
                num_rows += 1

            # Calculate the step size for latitude and longitude
            lat_step = (lat_max - lat_min) / num_rows
            lon_step = (lon_max - lon_min) / num_cols

            # Generate the zones
            zone_id_generator = utilities.IdGenerator()
            self.zones = []
            for row in range(num_rows):
                for col in range(num_cols):
                    zone_lat_min = lat_min + row * lat_step
                    zone_lat_max = zone_lat_min + lat_step
                    zone_lon_min = lon_min + col * lon_step
                    zone_lon_max = zone_lon_min + lon_step

                    # Ensure the last zone doesn't exceed the map boundaries
                    zone_lat_max = min(zone_lat_max, lat_max)
                    zone_lon_max = min(zone_lon_max, lon_max)

                    self.zones.append(
                        Zone(
                            zone_id=zone_id_generator.next(), 
                            lat_min=zone_lat_min, 
                            lat_max=zone_lat_max, 
                            lon_min=zone_lon_min, 
                            lon_max=zone_lon_max,
                            alpha=self.forecast_recency_bias,
                        )
                    )
            
    def get_zone(self, location):
        try: 
            return next(zone for zone in self.zones if zone.contains(location))
        except StopIteration:
            raise ValueError(f"Location(latitude={location.latitude}, longitude={location.longitude}) not in any zone")

    def generate_future_requests(self):

        index = self.intensity_function_index_generator.generate()
        self.intensity_function = self.intensity_functions[index]
        self.lambda_max = self.max_intensities[index]
        self.parent_interarrival_time_generator.parameters = {"scale": 1 / self.lambda_max}
        self.log(self, f"Max mean intensity: {self.lambda_max}")

        if self.verbose:
            plt.plot(self.domain, self.intensity_function(self.domain))
            plt.title(f"Intensity Function Sampled from Gaussian Process Prior")
            plt.xlabel("t")
            plt.ylabel("λ")
            plt.grid()
            plt.show()

        arrival_time = 0
        parent_requests = []
        parent_arrival_times = []
        self.num_casualties_generated = 0
        while arrival_time < self.time_limit:
            # Generate a candidate arrival time
            arrival_time += self.parent_interarrival_time_generator.generate()
            if arrival_time > self.time_limit:
                break

            # Evaluate the intensity funtion at the candidate arrival time
            if self.acceptance_probability_generator.generate() > self.intensity_function(arrival_time) / self.lambda_max:
                continue
            casualty_cluster_center = self.casualty_cluster_center_generator.generate()
            km_radius = 25 # 25 km -> 15.5 miles
            km_per_deg_lat = 111.32 # There are ~111.32 km in one degree of latitude
            latitudinal_std = km_radius / km_per_deg_lat
            longitudinal_std = km_radius / (km_per_deg_lat * math.cos(math.radians(casualty_cluster_center.latitude))) # 11 km (degrees longitude depends on latitude)
            latitude=scipy.stats.truncnorm.rvs(
                    a=(self.map["lat_min"] - casualty_cluster_center.latitude) / latitudinal_std, # lower bound; (a - loc) / scale = (a - mean) / std
                    b=(self.map["lat_max"] - casualty_cluster_center.latitude) / latitudinal_std, # upper bound
                    loc=casualty_cluster_center.latitude, # mean
                    scale=latitudinal_std,
                    random_state=self.pickup_site_rng,
                )
            # clip latitude to map boundaries
            if latitude < self.map["lat_min"]:
                latitude = self.map["lat_min"]
            if latitude > self.map["lat_max"]:
                latitude = self.map["lat_max"]
            longitude=scipy.stats.truncnorm.rvs(
                a=(self.map["lon_min"] - casualty_cluster_center.longitude) / longitudinal_std, # lower bound; (a - loc) / scale = (a - mean) / std
                b=(self.map["lon_max"] - casualty_cluster_center.longitude) / longitudinal_std, # upper bound
                loc=casualty_cluster_center.longitude, # mean
                scale=longitudinal_std,
                random_state=self.pickup_site_rng,
                )
            # clip longitude to map boundaries
            if longitude < self.map["lon_min"]:
                longitude = self.map["lon_min"]
            if longitude > self.map["lon_max"]:
                longitude = self.map["lon_max"]
            pickup_site = Location(
                latitude=latitude, 
                longitude=longitude,
            )
            origin_zone = self.get_zone(pickup_site)
            num_patients = self.num_patients_generator.generate() 
            num_routine = 0
            num_priority = 0
            num_urgent = 0
            for _ in range(num_patients):
                rand = self.precedence_probability_generator.generate()
                if rand <= 0.45: # 45% chance of routine
                    num_routine += 1
                elif rand <= 0.45 + 0.44: # 44% chance of priority
                    num_priority += 1
                else: # 11% chance of urgent
                    num_urgent += 1
            
            parent_requests.append(
                Request(
                    None, 
                    arrival_time, 
                    origin_zone, 
                    pickup_site, 
                    num_urgent, 
                    num_priority, 
                    num_routine
                )
            )

            parent_arrival_times.append(arrival_time)
            self.num_casualties_generated += num_patients

        descendent_requests = []
        descendent_arrival_times = []
        excitation_factor = 0.4 # alpha_0
        decay_rate = 1 # beta_0
        branching_ratio = excitation_factor / decay_rate

        for parent_request in parent_requests:
            num_descendents = self.child_count_generator.generate()
            for _ in range(num_descendents):
                # Generate the arrival times of the descendent requests
                descendent_arrival_time = parent_request.arrival_time + self.child_interarrival_time_generator.generate()
                if descendent_arrival_time < self.time_limit:
                    pickup_site = parent_request.pickup_site.copy()
                    origin_zone = self.get_zone(pickup_site)
                    num_patients = self.num_patients_generator.generate()
                    num_urgent = 0
                    num_priority = 0
                    num_routine = 0
                    for _ in range(num_patients):
                        rand = self.precedence_probability_generator.generate()
                        if rand <= 0.45: # 45% chance of urgent
                            num_urgent += 1
                        elif rand <= 0.45 + 0.44: # 44% chance of priority
                            num_priority += 1
                        else: # 11% chance of routine
                            num_routine += 1
                    descendent_requests.append(
                        Request(
                            None,
                            descendent_arrival_time,
                            origin_zone,
                            pickup_site,
                            num_urgent,
                            num_priority,
                            num_routine
                        )
                    )
                    descendent_arrival_times.append(descendent_arrival_time)
                    self.num_casualties_generated += num_patients

        all_requests = sorted(parent_requests + descendent_requests, key = lambda request: request.arrival_time)
        all_arrival_times = sorted(parent_arrival_times + descendent_arrival_times)
        self.num_parent_requests_generated = len(parent_requests)
        self.num_descendent_requests_generated = len(descendent_requests)
        self.total_num_requests_generated = len(all_requests)
        for request in all_requests:
            request.id = self.request_id_generator.next()
            self.schedule_event(
                Event(
                    event_type=EventType.RECEIVED_REQUEST, 
                    handler=self.request_manager, 
                    time=request.arrival_time, 
                    data={"request": request}
                )    
            )
        self.log(self, f"{self.num_casualties_generated} total casualties generated")
        self.log(self, f"{self.total_num_requests_generated} total requests generated")
        self.log(self, f"{self.num_parent_requests_generated} parent requests generated")
        self.log(self, f"{self.num_descendent_requests_generated} descendent requests generated")
        if self.verbose:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlim(self.map["lon_min"], self.map["lon_max"])
            ax.set_ylim(self.map["lat_min"], self.map["lat_max"])
            ax.set_title('Parent Request Arrival Locations')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True)
            color = {
                Request.Precedence.URGENT: 'red',
                Request.Precedence.PRIORITY: 'orange',
                Request.Precedence.ROUTINE: 'purple'
            }
            for request in parent_requests:
                ax.plot(request.pickup_site.longitude, request.pickup_site.latitude, 'o', color=color[request.precedence])
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.hist(parent_arrival_times, bins=self.time_limit, edgecolor='black')
            plt.xlabel('Time')
            plt.ylabel('Number of Arrivals')
            plt.title('Histogram of Parent Arrival Times')
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.hist(descendent_arrival_times, bins=self.time_limit, edgecolor='black')
            plt.xlabel('Time')
            plt.ylabel('Number of Arrivals')
            plt.title('Histogram of Descendent Arrival Times')
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.hist(all_arrival_times, bins=self.time_limit, edgecolor='black')
            plt.xlabel('Time')
            plt.ylabel('Number of Arrivals')
            plt.title('Histogram of All Arrival Times')
            plt.grid(True)
            plt.show()

    """
    1. Rerouted units can only be rerouted to requests with higher precedence.
    2. Each request can only be assigned to an MTF that posseses the required capabilities 
        - Role II can treat routine and priority
        - Role III can treat routine, priority, and urgent
    3. Each request can only be assigned to an MTF with sufficient bed capacity
    4. Each helicopter can only be assigned to a request if it has sufficient fuel
        - Must travel from its current location to the pickup_site, then to the MTF, and finally to the staging area
    """
    def get_feasible_decision_info(self, request):
        mission_distances = []
        mask = np.array([False for _ in self.decision_space])

        for index, (helicopter, MTF, staging_area) in self.decision_index_map.items():
            if not helicopter: # null decision, i.e., no assignment
                mission_distances.append((index, np.inf))
                mask[index] = True
                continue

            if helicopter.status.is_reroutable and request.precedence > helicopter.request.precedence:
                available_beds = MTF.num_available_beds + MTF.get_num_reserved_beds(helicopter.request)
            elif helicopter.status.is_dispatchable:
                available_beds = MTF.num_available_beds
            else:
                continue

            if MTF.role >= request.required_MTF_role and available_beds >= request.num_patients:
                mission_distance = helicopter.location.distance_to(request.pickup_site) + request.pickup_site.distance_to(MTF.location)
                total_distance = mission_distance + MTF.location.distance_to(staging_area.location)
                
                if helicopter.distance_to_empty >= total_distance:
                    mission_distances.append((index, mission_distance))
                    mask[index] = True

        mission_distances.sort(key=lambda x: x[1])

        self.log(self, f"Calculated feasible decision indicies: {mission_distances}")

        return {"mission_distances": mission_distances, "mask": mask}

    def get_reward(self):
        return self.reward

    def process_next_event(self):
        try:
            event = self.future_events.pop_event()

            if event.time > self.time_limit:
                self.log(self, "Time limit reached")
                return None

            self.previous_event_time = self.time
            self.time = event.time
            self.log(self, f"Processing event: {event}")
            event.process()
            self.log(self, f"Processed event: {event}")
            self.previous_event = event

            self.log(self, "Updating state")
            self.update_state()
            self.log(self, "State updated")

            if self.verbose:
                self.to_table()

            return event 
        except IndexError:
            self.log(self, "No more events in queue")
            return None

    def step(self, decision_index):
        self.num_steps += 1
        self.log(self, f"**************************Beginning step {self.num_steps}*****************************")
        self.reward = 0 # base reward for each step
        decision = self.decision_index_map[decision_index]
        self.implement_decision(self.request_to_assign, decision)

        while request := self.request_manager.get_next_request_to_examine():
            self.log(self, f"Examining next queued request ({request})")
            feasible_decision_info = self.get_feasible_decision_info(request)
            if len(feasible_decision_info["mission_distances"]) > 1: # there is more than one feasible decision
                self.request_to_assign = request
                self.log(self, f"Request to assign: {request}")
                state = self.get_observation(feasible_decision_info)
                reward = self.get_reward()
                self.log(self, f"Reward: {reward}")    
                terminated = False
                truncated = False
                self.log(self, f"***************************Ending step {self.num_steps}**************************")
                return state, reward, terminated, truncated, feasible_decision_info
            self.request_manager.add_request_to_revisit(request)
            self.log(self, f"No feasible decisions for {request}, continuing to next unexamined request")
        self.log(self, "No requests to examine, continuing to next event")

        while event := self.process_next_event():
            if event.type in {
                EventType.RECEIVED_REQUEST,
                EventType.OFFLOADED_PATIENTS,
                EventType.COMPLETED_REFUELING,
                EventType.COMPLETED_PATIENT_TREATMENT,
            }: # only review queued requests after events that can lead to new feasible decisions
                self.request_manager.reset_unexamined_requests()
                while request := self.request_manager.get_next_request_to_examine():
                    self.log(self, f"Examining next queued request ({request})")
                    feasible_decision_info = self.get_feasible_decision_info(request)
                    if len(feasible_decision_info["mission_distances"]) > 1: # there is more than one feasible decision
                        self.request_to_assign = request
                        self.log(self, f"Request to assign: {request}")
                        state = self.get_observation(feasible_decision_info)
                        reward = self.get_reward()
                        self.log(self, f"Reward: {reward}")    
                        terminated = False
                        truncated = False
                        self.log(self, f"******************************Ending step {self.num_steps}*******************************")
                        return state, reward, terminated, truncated, feasible_decision_info
                    self.request_manager.add_request_to_revisit(request)
                    self.log(self, f"No feasible decisions for {request}, continuing to next unexamined request")
            self.log(self, "No requests to examine, continuing to next event")

        self.log(self, "No more events to process")
        self.request_to_assign = None
        state = self.get_observation(None)
        feasible_decision_info = None
        reward = 0
        terminated = True
        truncated = False
        return state, reward, terminated, truncated, feasible_decision_info

    def implement_decision(self, request, decision):
        helicopter = decision[0]
        MTF = decision[1]
        staging_area = decision[2]
        if helicopter:
            self.schedule_event(
                Event(
                    event_type=EventType.RECEIVED_DECISION, 
                    handler=helicopter, 
                    time=self.time, 
                    data={"request": request, "MTF": MTF, "staging_area": staging_area}
                )
            )
            self.log(self, f"Implementing decision: ({request}, {helicopter}, {MTF}, {staging_area})")
            decision_event = self.process_next_event() # process the decision event           
        else:
            self.log(self, f"Implementing decision: ({request}, None, None, None)")
            self.request_manager.add_request_to_revisit(request)
            
    def to_table(self):
        # Convert individual entity lists to DataFrames
        requests_df = pd.DataFrame([
            request.to_dict() for request in (
            self.request_manager.requests_to_examine + 
            self.request_manager.requests_to_revisit +
            list(self.request_manager.requests_being_serviced.values())
            ) if request is not None
        ])
        helicopters_df = pd.DataFrame([helicopter.to_dict() for helicopter in self.helicopters])
        MTFs_df = pd.DataFrame([MTF.to_dict() for MTF in self.MTFs])
        
        # Select relevant columns for each entity type
        if not requests_df.empty:
            requests_df = requests_df[["id", "arrival_time", "precedence", "deadline", "hours_past_deadline", "hours_waiting"]]
        if not helicopters_df.empty:
            helicopters_df = helicopters_df[["id", "status", "request", "MTF", "staging_area", "distance_to_empty"]]
        if not MTFs_df.empty:
            MTFs_df = MTFs_df[["id", "role", "num_available_beds"]]

        # Store dataframes in a dictionary for organization
        grouped_data = {
            "Requests": requests_df,
            "Helicopters": helicopters_df,
            "MTFs": MTFs_df,
        }

        # Display or process each table
        for entity_type, df in grouped_data.items():
            print(f"{'='*20} {entity_type} {'='*20}\n")
            if not df.empty:
                print(df.to_string(index=False))
            else:
                print(f"No data available for {entity_type}.")
            print("\n")

    def display_metrics(self):
        staging_area_metrics = pd.DataFrame([{staging_area.id: staging_area.get_service_metrics()} for staging_area in self.staging_areas])
        overall_request_metrics = pd.DataFrame([self.request_manager.get_service_metrics()])
        zone_metrics = pd.DataFrame([{zone.id: zone.get_service_metrics()} for zone in self.zones])

        print("Staging Area Metrics:")
        print(staging_area_metrics.to_string(index=False))
        print("\nOverall Request Metrics:")
        print(overall_request_metrics.to_string(index=False))
        print("\nZone Metrics:")
        print(zone_metrics.to_string(index=False))


def random_policy(state, info, rng):
    return rng.choice(info["mission_distances"])[0]


def greedy_policy(state, info):
    return info["mission_distances"][0][0]