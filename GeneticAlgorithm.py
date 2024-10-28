import random
import copy
import math
from dataclasses import dataclass
from typing import List, Dict, Set


@dataclass
class Activity:
    name: str
    enrollment: int
    preferred_facilitators: List[str]
    other_facilitators: List[str]


@dataclass
class Room:
    name: str
    capacity: int


@dataclass
class ScheduledActivity:
    room: Room
    time: str
    facilitator: str
    activity: Activity


class SchedulingGA:
    def __init__(self, population_size=500, mutation_rate=0.01):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.initialize_data()

    #softmax function instead of using scipy
    def softmax(self, x):
        exp_x = [math.exp(i) for i in x]
        sum_exp_x = sum(exp_x)
        return [j / sum_exp_x for j in exp_x]

    def initialize_data(self):
        # Initialize activities
        self.activities = {
            'SLA100A': Activity('SLA100A', 50,
                                ['Glen', 'Lock', 'Banks', 'Zeldin'],
                                ['Numen', 'Richards']),
            'SLA100B': Activity('SLA100B', 50,
                                ['Glen', 'Lock', 'Banks', 'Zeldin'],
                                ['Numen', 'Richards']),
            'SLA191A': Activity('SLA191A', 50,
                                ['Glen', 'Lock', 'Banks', 'Zeldin'],
                                ['Numen', 'Richards']),
            'SLA191B': Activity('SLA191B', 50,
                                ['Glen', 'Lock', 'Banks', 'Zeldin'],
                                ['Numen', 'Richards']),
            'SLA201': Activity('SLA201', 50,
                               ['Glen', 'Banks', 'Zeldin', 'Shaw'],
                               ['Numen', 'Richards', 'Singer']),
            'SLA291': Activity('SLA291', 50,
                               ['Lock', 'Banks', 'Zeldin', 'Singer'],
                               ['Numen', 'Richards', 'Shaw', 'Tyler']),
            'SLA303': Activity('SLA303', 60,
                               ['Glen', 'Zeldin', 'Banks'],
                               ['Numen', 'Singer', 'Shaw']),
            'SLA304': Activity('SLA304', 25,
                               ['Glen', 'Banks', 'Tyler'],
                               ['Numen', 'Singer', 'Shaw', 'Richards', 'Uther', 'Zeldin']),
            'SLA394': Activity('SLA394', 20,
                               ['Tyler', 'Singer'],
                               ['Richards', 'Zeldin']),
            'SLA449': Activity('SLA449', 60,
                               ['Tyler', 'Singer', 'Shaw'],
                               ['Zeldin', 'Uther']),
            'SLA451': Activity('SLA451', 100,
                               ['Tyler', 'Singer', 'Shaw'],
                               ['Zeldin', 'Uther', 'Richards', 'Banks'])
        }

        # Initialize rooms
        self.rooms = [
            Room('Slater 003', 45),
            Room('Roman 216', 30),
            Room('Loft 206', 75),
            Room('Roman 201', 50),
            Room('Loft 310', 108),
            Room('Beach 201', 60),
            Room('Beach 301', 75),
            Room('Logos 325', 450),
            Room('Frank 119', 60)
        ]

        # Initialize times
        self.times = ['10 AM', '11 AM', '12 PM', '1 PM', '2 PM', '3 PM']

        # Initialize facilitators
        self.facilitators = [
            'Lock', 'Glen', 'Banks', 'Richards', 'Shaw', 'Singer',
            'Uther', 'Tyler', 'Numen', 'Zeldin'
        ]

    def create_random_schedule(self) -> List[ScheduledActivity]:
        schedule = []
        for activity in self.activities.values():
            schedule.append(ScheduledActivity(
                room=random.choice(self.rooms),
                time=random.choice(self.times),
                facilitator=random.choice(self.facilitators),
                activity=activity
            ))
        return schedule

    def calculate_fitness(self, schedule: List[ScheduledActivity]) -> float:
        fitness = 0

        # Create helper dictionaries for quick lookups
        time_room_pairs = {}
        facilitator_times = {}
        facilitator_count = {}
        facilitator_activities_per_time = {}

        # Initialize facilitator activities per time slot dictionary
        for facilitator in self.facilitators:
            facilitator_activities_per_time[facilitator] = {}
            for time in self.times:
                facilitator_activities_per_time[facilitator][time] = 0

        for s in schedule:

            # Count facilitator assignments per time slot
            facilitator_activities_per_time[s.facilitator][s.time] += 1

            # Count facilitator assignments
            facilitator_count[s.facilitator] = facilitator_count.get(s.facilitator, 0) + 1

            # Track facilitator times
            if s.facilitator not in facilitator_times:
                facilitator_times[s.facilitator] = []
            facilitator_times[s.facilitator].append(s.time)

            # Activity is scheduled at the same time in the same room as another of the activities: -0.5
            time_room_key = (s.time, s.room.name)
            if time_room_key in time_room_pairs:
                fitness -= 0.5
            time_room_pairs[time_room_key] = s.activity.name

            # Room size checks
            # Activities is in a room too small for its expected enrollment: -0.5
            # Activities is in a room with capacity > 3 times expected enrollment: -0.2
            # Activities is in a room with capacity > 6 times expected enrollment: -0.4
            # Otherwise + 0.3
            if s.room.capacity < s.activity.enrollment:
                fitness -= 0.5
            elif s.room.capacity > 3 * s.activity.enrollment:
                fitness -= 0.2
                if s.room.capacity > 6 * s.activity.enrollment:
                    fitness -= 0.2
            else:
                fitness += 0.3

            # Facilitator preference checks
            # Activities is overseen by a preferred facilitator: + 0.5
            # Activities is overseen by another facilitator listed for that activity: +0.2
            # Activities is overseen by some other facilitator: -0.1
            if s.facilitator in s.activity.preferred_facilitators:
                fitness += 0.5
            elif s.facilitator in s.activity.other_facilitators:
                fitness += 0.2
            else:
                fitness -= 0.1

            # Special rules for SLA101 and SLA191 sections
            # The 2 sections of SLA 101 are more than 4 hours apart: + 0.5
            # Both sections of SLA 101 are in the same time slot: -0.5
            # The 2 sections of SLA 191 are more than 4 hours apart: + 0.5
            # Both sections of SLA 191 are in the same time slot: -0.5
            if s.activity.name in ['SLA100A', 'SLA100B', 'SLA191A', 'SLA191B']:
                partner_section = None
                if s.activity.name == 'SLA100A':
                    partner_section = next(x for x in schedule if x.activity.name == 'SLA100B')
                elif s.activity.name == 'SLA100B':
                    partner_section = next(x for x in schedule if x.activity.name == 'SLA100A')
                elif s.activity.name == 'SLA191A':
                    partner_section = next(x for x in schedule if x.activity.name == 'SLA191B')
                elif s.activity.name == 'SLA191B':
                    partner_section = next(x for x in schedule if x.activity.name == 'SLA191A')

                if partner_section:
                    time_diff = abs(self.times.index(s.time) - self.times.index(partner_section.time))
                    if time_diff > 4:
                        fitness += 0.5
                    elif time_diff == 0:
                        fitness -= 0.5

            # Check for consecutive slots between SLA191 and SLA101
            # A section of SLA 191 and a section of SLA 101 are overseen in consecutive time slots
            # (e.g., 10 AM & 11 AM): +0.5
            # In this case only (consecutive time slots),
            # one of the activities is in Roman or Beach, and the other isn’t: -0.4
            # It’s fine if neither is in one of those buildings, of activity;
            # we just want to avoid having consecutive activities being widely separated.
            # A section of SLA 191 and a section of SLA 101 are taught separated by 1 hour
            # (e.g., 10 AM & 12:00 Noon): + 0.25
            # A section of SLA 191 and a section of SLA 101 are taught in the same time slot: -0.25
            if s.activity.name.startswith('SLA191'):
                for other in schedule:
                    if other.activity.name.startswith('SLA100'):
                        time_diff = abs(self.times.index(s.time) - self.times.index(other.time))
                        if time_diff == 1:  # Consecutive slots
                            fitness += 0.5
                            # Check if they're in different buildings when consecutive
                            if ((s.room.name.startswith(('Roman', 'Beach')) and
                                 not other.room.name.startswith(('Roman', 'Beach'))) or
                                    (not s.room.name.startswith(('Roman', 'Beach')) and
                                     other.room.name.startswith(('Roman', 'Beach')))):
                                fitness -= 0.4
                        elif time_diff == 2:  # Separated by 1 hour
                            fitness += 0.25
                        elif time_diff == 0:  # Same time slot
                            fitness -= 0.25

        # Facilitator load checks
        # Activity facilitator is scheduled for only 1 activity in this time slot: + 0.2
        # Activity facilitator is scheduled for more than one activity at the same time: - 0.2

        # Second loop: Check facilitator workload after we have all counts
        for s in schedule:
            activities_this_slot = facilitator_activities_per_time[s.facilitator][s.time]
            if activities_this_slot == 1:
                fitness += 0.2  # Bonus for single activity in time slot
            elif activities_this_slot > 1:
                fitness -= 0.2  # Penalty for multiple activities in same time slot

        # Facilitator is scheduled to oversee more than 4 activities total: -0.5
        # Facilitator is scheduled to oversee 1 or 2 activities*: -0.4
        # Exception: Dr. Tyler is committee chair and has other demands on his time.
        # No penalty if he’s only required to oversee < 2 activities.
        # If any facilitator scheduled for consecutive time slots:
        for facilitator, count in facilitator_count.items():
            if count > 4:
                fitness -= 0.5
            elif count < 3 and facilitator != 'Tyler':
                fitness -= 0.4
            # Check for consecutive time slots
            times = sorted(facilitator_times[facilitator], key=lambda x: self.times.index(x))
            for i in range(len(times) - 1):
                if abs(self.times.index(times[i]) - self.times.index(times[i + 1])) == 1:
                    fitness -= 0.2

        return fitness

    # Crossover function to form child schedules
    def crossover(self, parent1: List[ScheduledActivity], parent2: List[ScheduledActivity]) -> List[ScheduledActivity]:
        if len(parent1) != len(parent2):
            raise ValueError("Parents must be of same length")

        crossover_point = random.randint(1, len(parent1) - 1)
        child = copy.deepcopy(parent1[:crossover_point] + parent2[crossover_point:])
        return child

    # Mutation function to introduce new data
    def mutate(self, schedule: List[ScheduledActivity]):
        for s in schedule:
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(['room', 'time', 'facilitator'])
                if mutation_type == 'room':
                    s.room = random.choice(self.rooms)
                elif mutation_type == 'time':
                    s.time = random.choice(self.times)
                else:
                    s.facilitator = random.choice(self.facilitators)

    # Function that runs the algorithm and manages selection
    def run(self, generations: int = 100):
        # Create initial population
        population = [self.create_random_schedule() for _ in range(self.population_size)]
        best_fitness_history = []
        avg_fitness_history = []

        last_avg_fitness = None

        for gen in range(generations):
            # Calculate fitness for all schedules and sort them
            population = sorted(population, key=lambda x: self.calculate_fitness(x), reverse=True)
            fitness_scores = [self.calculate_fitness(schedule) for schedule in population]

            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            # Dynamic mutation rate adjustment
            if last_avg_fitness is not None and avg_fitness > last_avg_fitness:
                self.mutation_rate /= 2  # Halve the mutation rate if fitness improved
                print(f"Mutation rate adjusted to: {self.mutation_rate:.8f}")

            last_avg_fitness = avg_fitness

            top_half = sorted(population, key=self.calculate_fitness, reverse=True)[:len(population) // 2]
            top_half_fitness_scores = [self.calculate_fitness(schedule) for schedule in top_half]
            probabilities = self.softmax(top_half_fitness_scores)

            # Generate the other half through crossover and mutation
            new_population = top_half[:]
            while len(new_population) < self.population_size:
                # Select two parents based on their probabilities
                parent_indices = random.choices(range(len(top_half)), weights=probabilities, k=2)
                parent1 = top_half[parent_indices[0]]
                parent2 = top_half[parent_indices[1]]

                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            population = new_population

            # Check for convergence after 100 generations
            if gen >= 100:
                improvement = (avg_fitness - avg_fitness_history[-100]) / abs(avg_fitness_history[-100])
                if improvement < 0.01:
                    print(f"Converged after {gen} generations")
                    break

            if gen % 10 == 0:
                print(f"Generation {gen}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")

        # Return best schedule
        best_schedule = max(population, key=lambda x: self.calculate_fitness(x))
        return best_schedule, best_fitness_history, avg_fitness_history

    # Prints the best schedule
    def print_schedule(self, schedule: List[ScheduledActivity], file_name: str = "schedule.txt"):
        # Sort schedule by time and room for better readability
        sorted_schedule = sorted(schedule, key=lambda x: (self.times.index(x.time), x.room.name))

        output = "\nFinal ScheduledActivity:\n"
        output += "=" * 80 + "\n"
        current_time = None

        for s in sorted_schedule:
            if s.time != current_time:
                current_time = s.time
                output += f"\nTime Slot: {current_time}\n"
                output += "-" * 80 + "\n"

            output += f"Activity: {s.activity.name:8} | Room: {s.room.name:12} | Facilitator: {s.facilitator:8}\n"

        output += f"\nFitness Score: {self.calculate_fitness(schedule)}\n"

        # Print to console
        print(output)

        # Write to file
        with open(file_name, 'w') as file:
            file.write(output)


def main():
    # Initialize and run the genetic algorithm
    ga = SchedulingGA(population_size=500, mutation_rate=0.01)
    best_schedule, best_fitness_history, avg_fitness_history = ga.run(generations=200)

    # Print the final schedule
    ga.print_schedule(best_schedule)


if __name__ == "__main__":
    main()