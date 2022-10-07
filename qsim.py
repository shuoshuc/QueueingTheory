#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import default_rng
import pandas as pd
from enum import Enum

NUM_ITER = 10

class EventType(Enum):
    ARRIVAL = 1
    COMPLETION =2

class QSim:
    def __init__(self):
        self._rng = default_rng()

        self.num_jobs = 0
        self.clock = 0.0
        self.t_next_arrival = self.clock + self._genIAT()
        self.t_next_completion = float('inf') 

        # Registers event handlers.
        self._event_handlers = {
            EventType.ARRIVAL: self._arrivalHelper,
            EventType.COMPLETION: self._completionHelper
        }

    def _genIAT(self):
        '''
        Generates an inter-arrival time (IAT). It is a duration, not absolute
        time.
        Note: assuming M/M/1 queue, IAT ~ Exp(1/scale).
        '''
        return self._rng.exponential(scale=10)

    def _genServiceTime(self):
        '''
        Generates the service time for a job.
        Note: assuming M/M/1 queue, service time ~ Exp(1/scale).
        '''
        return self._rng.exponential(scale=10)

    def _eventHandler(self, event_type):
        '''
        Invokes the corresponding event handler.
        '''
        return self._event_handlers[event_type]()
        
    def _arrivalHelper(self):
        '''
        Updates states in response to an arrival event.
        '''
        self.num_jobs += 1
        # Only generates a service time when there is only 1 job in system. This
        # gets serviced immediately.
        if self.num_jobs == 1:
            service_time = self._genServiceTime()
            self.t_next_completion = self.clock + service_time
        self.t_next_arrival = self.clock + self._genIAT()
        
    def _completionHelper(self):
        '''
        Updates states in response to a completion event.
        '''
        self.num_jobs -= 1
        if self.num_jobs > 0:
            service_time = self._genServiceTime();
            self.t_next_completion = self.clock + service_time
        else:
            self.t_next_completion = float("inf")
    
    def run(self):
        '''
        Advances the simulation by one event.
        '''
        # next_event is a tuple where first is event type and second is time.
        next_event = min((EventType.ARRIVAL, self.t_next_arrival),
                         (EventType.COMPLETION, self.t_next_completion),
                         key=lambda x: x[1])
        # Advances clock to the next event.
        self.clock = next_event[1]
        # Updates states given the event.
        self._eventHandler(next_event[0])


if __name__ == "__main__":
    qsim = QSim()

    print(f"[Start] clock={qsim.clock}, # jobs: {qsim.num_jobs}, next arrival: " 
          f"{qsim.t_next_arrival}, next completion: {qsim.t_next_completion}")

    for i in range(NUM_ITER):
        qsim.run()
        njobs = qsim.num_jobs
        clock = qsim.clock
        arrival = qsim.t_next_arrival
        completion = qsim.t_next_completion
        print(f"[Iter {i}] clock={clock}, # jobs: {njobs}, next arrival: "
              f"{arrival}, next completion: {completion}")
