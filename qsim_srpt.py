#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from numpy.random import default_rng
from enum import Enum
import queue, heapq
import threading
import numpy as np

NUM_ITER = 1000000

class EventType(Enum):
    ARRIVAL = 1
    COMPLETION =2

class AtomicCounter:
    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value

@dataclass(order=True)
class Job:
    sort_index: int = field(init=False)
    jid: int
    t_arrival: float
    service_time: float

    def __post_init__(self):
           self.sort_index = self.service_time

class JobQueue:
    '''
    Unbounded PS job queue implementation.
    '''
    def __init__(self, current_time):
        self.q = []
        self.q_clock = current_time

    def enqueue(self, job):
        if len(self.q):
            top_job = heapq.heappop(self.q)
            # Update first job's service time.
            top_job.service_time -= job.t_arrival - self.q_clock
            heapq.heappush(self.q, top_job)
        # New job's arrival time is the current clock.
        self.q_clock = job.t_arrival
        heapq.heappush(self.q, job)

    def dequeue(self, current_time):
        completed_job = heapq.heappop(self.q)
        # Updates clock as job leaves queue.
        self.q_clock = current_time
        return completed_job

    def peek(self):
        if len(self.q):
            # Smallest job is the first.
            return self.q[0]
        else:
            return None

    def qlen(self):
        return len(self.q)

class QSim:
    def __init__(self):
        self._rng = default_rng()
        self.num_jobs = 0
        self.clock = 0.0
        self.t_next_arrival = self.clock + self._genIAT()
        self.t_next_completion = float('inf') 

        # A SRPT queue tracking the arrival times of each job, for computing the
        # *response* time later.
        self.q = JobQueue(self.clock)
        self.total_response_time = 0
        # Number of jobs fully completed.
        self.served_jobs = 0

        self.global_jid = AtomicCounter()
        # Job id of the current job under service.
        self.curr_jid = -1

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
        '''
        return self._rng.exponential(scale=1/0.8)
        '''
        rho = 0.9
        return self._rng.exponential(scale=(2999.99/rho))

    def _genServiceTime(self):
        '''
        Generates the service time for a job.
        Note: assuming M/G/1 queue, service time ~ DegenerateHyperExp(mu=1, p).
        '''
        '''
        p = 1/4
        # With probability 1 - p, service time is 0.
        if self._rng.uniform(low=0, high=1) > p:
            return 0
        return self._rng.exponential(scale=1/p)
        '''
        # Bounded Pareto distribution
        a, k, p = 1.8, 1333.33, 1e11
        s = (self._rng.pareto(a) + 1) * k
        return s / (1 - np.power(k/p, a)) if s > 7.58861e-23 else 0

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
        self.q.enqueue(Job(self.global_jid.increment(), self.clock, self._genServiceTime()))
        # Find out which job has the shortest remaining processing time.
        job = self.q.peek()
        self.curr_jid = job.jid
        self.t_next_completion = self.clock + job.service_time
        self.t_next_arrival = self.clock + self._genIAT()
        
    def _completionHelper(self):
        '''
        Updates states in response to a completion event.
        '''
        self.num_jobs -= 1
        completed_job = self.q.dequeue(self.clock)
        self.total_response_time += self.clock - completed_job.t_arrival
        self.served_jobs += 1
        if self.num_jobs > 0:
            # Continue serving the rest jobs.
            next_job = self.q.peek()
            self.curr_jid = next_job.jid
            self.t_next_completion = self.clock + next_job.service_time
        else:
            self.curr_jid = -1
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
        if qsim.served_jobs:
            ET = qsim.total_response_time / qsim.served_jobs
        else:
            ET = None
        print(f"[Iter {i}] clock={qsim.clock}, # jobs: {qsim.num_jobs}, "
              f"next arrival: {qsim.t_next_arrival}, next completion: "
              f"{qsim.t_next_completion}, current job: {qsim.curr_jid}, "
              f"E[T]={ET}")
