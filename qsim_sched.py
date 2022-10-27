#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import default_rng
from enum import Enum
import queue

NUM_ITER = 1000000

class EventType(Enum):
    ARRIVAL = 1
    COMPLETION =2

class QSim:
    def __init__(self):
        self._rng = default_rng()
        self.num_jobs = 0
        # A served job is either completed or currently under service.
        self.served_jobs = 0
        self.clock = 0.0
        self.t_next_arrival = self.clock + self._genIAT()
        self.t_next_completion = float('inf') 
        # A FIFO queue tracking the arrival times of each job, for computing the
        # queueing time later.
        self.q = queue.SimpleQueue()
        # A FIFO queue tracking the arrival times of each job, for computing the
        # *response* time later.
        self.q2 = queue.SimpleQueue()
        self.total_queueing_time = 0
        self.total_response_time = 0

        # Registers event handlers.
        self._event_handlers = {
            EventType.ARRIVAL: self._arrivalHelper,
            EventType.COMPLETION: self._completionHelper
        }

    def _genIAT(self):
        '''
        Generates an inter-arrival time (IAT). It is a duration, not absolute
        time.
        Note: assuming M/G/1 queue, IAT ~ Exp(1/scale).
        '''
        return self._rng.exponential(scale=1/0.8)

    def _genServiceTime(self):
        '''
        Generates the service time for a job.
        Note: assuming M/G/1 queue, service time ~ DegenerateHyperExp(mu=1, p).
        '''
        p = 1/5
        # With probability 1 - p, service time is 0.
        if self._rng.uniform(low=0, high=1) > p:
            return 0
        return self._rng.exponential(scale=1/p)

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
        self.q2.put(self.clock)
        # Only generates a service time when there is only 1 job in system. This
        # gets serviced immediately.
        if self.num_jobs == 1:
            service_time = self._genServiceTime()
            self.t_next_completion = self.clock + service_time
            # Only 1 job in the queue, there is no queueing delay, consider the
            # job immediately served.
            self.served_jobs += 1
        else:
            # Enqueue current time for the job arrived. Only do this when there
            # is actual queueing.
            self.q.put(self.clock)
        self.t_next_arrival = self.clock + self._genIAT()
        
    def _completionHelper(self):
        '''
        Updates states in response to a completion event.
        '''
        self.num_jobs -= 1
        self.total_response_time += self.clock - self.q2.get()
        if self.num_jobs > 0:
            service_time = self._genServiceTime();
            self.t_next_completion = self.clock + service_time
            # Serves the next job in queue and computes queueing delay.
            self.served_jobs += 1
            self.total_queueing_time += self.clock - self.q.get()
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
        print(f"[Iter {i}] clock={qsim.clock}, # jobs: {qsim.num_jobs}, "
              f"next arrival: {qsim.t_next_arrival}, next completion: "
              f"{qsim.t_next_completion}, "
              f"E[T_Q]={qsim.total_queueing_time / qsim.served_jobs}, "
              f"E[T]={qsim.total_response_time / qsim.served_jobs}")
