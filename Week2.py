from enum import IntEnum
import random


class GameRules(IntEnum):
    INITIAL_HEALTH_POINTS = 1000
    INITIAL_ENERGY = 1000


class RobotPoints(IntEnum):
    FORCE = 5
    FAIL_PERCENTAGE = 10


class Robot:

    def __init__(self, name):

        self.name = name

        self.initial_energy = GameRules.INITIAL_ENERGY
        self.initial_health_points = GameRules.INITIAL_HEALTH_POINTS

        self.force = RobotPoints.FORCE
        self.fail_percentage = RobotPoints.FAIL_PERCENTAGE

        self.energy = self.initial_energy
        self.health_points = self.initial_health_points

    def decide_on_action(self, incoming_damage):
        # for now, robots will be very dumb and always attack.

        self.health_points -= incoming_damage

        if self.energy > 0:
            attack_strength = self.attack()
            return attack_strength
        else:
            return 0

    def attack(self):

        self.energy -= 1

        if random.randint(0, 100) < self.fail_percentage:
            return 0

        return self.force

    def reset(self):
        # fill this
        self.energy = self.initial_energy
        self.health_points = self.initial_health_points
        pass


class Battle:

    def __init__(self, robot_1, robot_2):

        self.robot_1 = robot_1
        self.robot_2 = robot_2

        # select first robot
        self.current_acting_robot = random.choice([self.robot_1, self.robot_2])

    def battle_ended(self):
        # returns True if the battle has finished
        if self.robot_1.health_points <= 0 or self.robot_2.health_points <= 0:
            return True
        if self.robot_1.energy <= 0 and self.robot_2.energy <= 0:
            return True

        return False

    def announce_winner(self):
        # Print the result of the battle, who is the winner with how many points
        # returns the winner
        if self.robot_1.health_points > self.robot_2.health_points:
            print(self.robot_1.name)
            print('energy:')
            print(self.robot_1.energy)
            print(self.robot_2.energy)
            print('HP:')
            print(self.robot_1.health_points)
            print(self.robot_2.health_points)
            return self.robot_1
        else:
            print(self.robot_2.name)
            print('energy:')
            print(self.robot_2.energy)
            print(self.robot_1.energy)
            print('HP:')
            print(self.robot_2.health_points)
            print(self.robot_1.health_points)
            return self.robot_2

    def battle_until_the_end(self):

        battle_ongoing = True

        # first action is initialized as an attack with 0 damage

        attack_points = 0

        while battle_ongoing:

            # inflict damages on the current robot
            # the current robot produces new_damages

            new_attack_points = self.current_acting_robot.decide_on_action(attack_points)

            # Change the current_acting_robot. If it was robot_1, then now it is robot_2
            if self.current_acting_robot == self.robot_1:
                self.current_acting_robot = self.robot_2
            else:
                self.current_acting_robot = self.robot_1

            # Check if battle ended
            if self.battle_ended() is True:
                battle_ongoing = False

            # Set the damage points sent by the previous robot.
            attack_points = new_attack_points

        winner = self.announce_winner()

        return winner


class StrongBot(Robot):

    def __init__(self, name, brutality):
        super().__init__(name)

        self.brutality = brutality

    def decide_on_action(self, incoming_damage):
        attack_strength = 0

        self.health_points -= incoming_damage

        if self.energy >= self.brutality / 2 + 1:
            attack_strength = self.attack()

        return attack_strength

    def attack(self):
        self.energy -= 1 + self.brutality * 0.5

        if random.randint(0, 100) < self.fail_percentage:
            return 0

        return self.force + self.brutality


class CleverBot(Robot):

    def __init__(self, name, intelligence):
        super().__init__(name)

        self.intelligence = intelligence
        self.dodge_percentage = 30 + 10 * intelligence
        if self.dodge_percentage > 100:
            self.dodge_percentage = 100

    def decide_on_action(self, incoming_damage):

        attack_strength = 0

        if self.energy > 0:

            if incoming_damage > 0:
                dodge_successful = self.dodge()
                if not dodge_successful:
                    self.health_points -= incoming_damage

            else:
                attack_strength = self.attack()

        return attack_strength

    def dodge(self):

        self.energy -= 2

        if random.randint(0, 99) < self.dodge_percentage:
            return True
        else:
            return False

# Now we can test it!
robot_1 = Robot('spock')
robot_2 = Robot('donald')

strong_bot = StrongBot('Paul', 1)
clever_bot = CleverBot('Gutza', 0)

battleground = Battle(clever_bot, strong_bot)
battleground.battle_until_the_end()
