#!/usr/bin/env python
# -*- coding: utf-8 -*-

import queue
import threading
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.random import default_rng

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
    job_type: int

    def __post_init__(self):
        '''
        Priority queue across buckets, FCFS within the same bucket.
        '''
        self.sort_index = self.job_type

class JobQueue:
    '''
    Unbounded FIFO job queue implementation.
    '''
    def __init__(self):
        self.q = []

    def enqueue(self, job):
        self.q.append(job)

    def dequeue(self):
        return self.q.pop(0)

    def peek(self):
        if len(self.q):
            # Smallest job is the first.
            return self.q[0]
        else:
            return None

    def qlen(self):
        return len(self.q)

class QSim:
    def __init__(self, rho):
        self.rho = rho
        self._rng = default_rng()
        self.num_jobs = 0
        self.clock = 0.0
        self.t_next_arrival = self.clock + self._genIAT()
        self.t_next_completion = float('inf') 
        self.job_type = -1

        # A FCFS queue tracking the arrival times of each job, for computing the
        # *response* time later.
        self.q = JobQueue()
        self.total_response_time_h = 0
        self.total_response_time_l = 0
        # Number of jobs fully completed.
        self.served_jobs_h = 0
        self.served_jobs_l = 0

        self.global_jid = AtomicCounter()
        # Job id of the current job under service.
        self.curr_jid = -1
        # Arrival time of the current job under service.
        self.t_curr_arrival = float('inf')

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
        return self._rng.exponential(scale=(16.2/self.rho))

    def _genServiceTime(self):
        '''
        Generates the service time for a job.
        Note: assuming M/G/1 queue, service time ~ DegenerateHyperExp(mu=1, p).
        '''
        # With prob 0.2, return topo size, else demand size.
        if self._rng.uniform(low=0, high=1) < 0.2:
            a, m = 2, 0.5
            s = (self._rng.pareto(a) + 1) * m
            return (s, 1)
        else:
            return (self._rng.exponential(scale=(1/0.05)), 1000000)

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
        service_time, job_type = self._genServiceTime()
        self.q.enqueue(Job(self.global_jid.increment(), self.clock,
                           service_time, job_type))
        if self.num_jobs == 1:
            job = self.q.dequeue()
            self.curr_jid = job.jid
            self.t_curr_arrival = job.t_arrival
            self.t_next_completion = self.clock + job.service_time
            self.job_type = job.job_type
        self.t_next_arrival = self.clock + self._genIAT()
        
    def _completionHelper(self):
        '''
        Updates states in response to a completion event.
        '''
        self.num_jobs -= 1
        if self.job_type > 1:
            self.total_response_time_l += self.clock - self.t_curr_arrival
            self.served_jobs_l += 1
        else:
            self.total_response_time_h += self.clock - self.t_curr_arrival
            self.served_jobs_h += 1
        if self.num_jobs > 0:
            # Continue serving the rest jobs.
            next_job = self.q.dequeue()
            self.curr_jid = next_job.jid
            self.t_curr_arrival = next_job.t_arrival
            self.t_next_completion = self.clock + next_job.service_time
            self.job_type = next_job.job_type
        else:
            self.curr_jid = -1
            self.t_curr_arrival = float('inf')
            self.t_next_completion = float('inf')
            self.job_type = -1
    
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
    for rho in [float(0.01 * i) for i in range(1, 100)]:
        qsim = QSim(rho)

        '''
        print(f"[Start] clock={qsim.clock}, # jobs: {qsim.num_jobs}, next arrival: " 
              f"{qsim.t_next_arrival}, next completion: {qsim.t_next_completion}")
        '''

        for i in range(NUM_ITER):
            qsim.run()
            if qsim.served_jobs_h:
                ETH = qsim.total_response_time_h / qsim.served_jobs_h
            else:
                ETH = None
            if qsim.served_jobs_l:
                ETL = qsim.total_response_time_l / qsim.served_jobs_l
            else:
                ETL = None
            '''
            print(f"[Iter {i}] clock={qsim.clock}, # jobs: {qsim.num_jobs}, "
                  f"next arrival: {qsim.t_next_arrival}, next completion: "
                  f"{qsim.t_next_completion}, current job: {qsim.curr_jid}, "
                  f"E[T]={ET}")
            '''
        print(f"{rho}, {ETH}, {ETL}")
