import numpy as np
from opponent_pool import Result

class GymAI(object):
    def __init__(self, gateway, pipe, env):
        self.gateway = gateway
        self.pipe = pipe
        self.env = env

        self.width = 96  # The width of the display to obtain
        self.height = 64  # The height of the display to obtain
        self.grayscale = (
            True  # The display's color to obtain true for grayscale, false for RGB
        )

        self.obs = None
        self.pre_framedata = None
        self.just_inited = True

        self._actions = "AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_UA AIR_UB BACK_JUMP BACK_STEP CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD DASH FOR_JUMP FORWARD_WALK JUMP NEUTRAL STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD THROW_A THROW_B"
        self.action_strs = self._actions.split(" ")

    def close(self):
        pass

    def initialize(self, gameData, player):
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()

        self.player = player
        self.gameData = gameData
        self.simulator = self.gameData.getSimulator()

        self.frozen_frames = 0
        return 0

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
        # print("send round end to {}".format(self.pipe))
        # tell gym the round ends
        if self.player:
            own_hp, opp_hp = x, y
        else:
            own_hp, opp_hp = y, x
        self.pipe.send([self.obs, 0, True, [own_hp, opp_hp]])
        if own_hp > opp_hp:
            strings = "you win"
        else:
            strings = "you lose"
        print("Fighting {}, At the end, own_hp {}: opp_hp {}. {}.".format(self.env.opponent, own_hp, opp_hp, strings), flush=True)
        self.env.opponent_pool.update_opponent(self.env.opponent, Result(own_hp, opp_hp))
        self.just_inited = True

        self.obs = None
        self.frozen_frames = 0
        self.pre_framedata = None

        self.inputKey.empty()
        self.cc.skillCancel()

    # Please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        self.screenData = sd

    def getInformation(self, frameData, isControl):
        self.frameData = frameData
        self.isControl = isControl
        self.cc.setFrameData(self.frameData, self.player)
        if frameData.getEmptyFlag():
            return

    def input(self):
        return self.inputKey

    def gameEnd(self):
        pass

    def processing(self):
        ## game starts but round not start
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingTime() <= 0:
            self.isGameJustStarted = True
            return

        if not self.isGameJustStarted:
            # totally start the round
            pass
            # Simulate the delay and look ahead 2 frames. The simulator class exists already in FightingICE
            # self.frameData = self.simulator.simulate(self.frameData, self.player, None, None, 17)
            # You can pass actions to the simulator by writing as follows:
            # actions = self.gateway.jvm.java.util.ArrayDeque()
            # actions.add(self.gateway.jvm.enumerate.Action.STAND_A)
            # self.frameData = self.simulator.simulate(self.frameData, self.player, actions, actions, 17)
        else:
            # If the game just started, no point on simulating
            self.isGameJustStarted = False

        ## wait for frozen frame (Neutral command)
        if self.frozen_frames > 0:
            self.frozen_frames -= 1
            return
        # continue unfinished commands
        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            return
        # wait for controllability
        if not self.isControl:
            return

        self.inputKey.empty()
        self.cc.skillCancel()

        ## prepare state for gym policy
        # if just inited, should wait for first reset()
        if self.just_inited:
            request = self.pipe.recv()
            if request == "reset":
                self.just_inited = False
                self.obs = self.get_obs(self.frameData, self.player)
                self.pre_framedata = self.frameData
                self.pipe.send(self.obs)
            else:
                raise ValueError
        # if not just inited but self.obs is none, it means second/thrid round just started
        # should return only obs for reset()
        elif self.obs is None:
            self.obs = self.get_obs(self.frameData, self.player)
            self.pre_framedata = self.frameData
            self.pipe.send(self.obs)
        # if there is self.obs, do step() and return [obs, reward, done, info]
        else:
            self.obs = self.get_obs(self.frameData, self.player)
            self.reward = self.get_reward(self.player)
            self.pre_framedata = self.frameData
            own_hp = self.frameData.getCharacter(self.player).getHp()
            opp_hp = self.frameData.getCharacter(not self.player).getHp()
            self.pipe.send([self.obs, self.reward, False, [own_hp, opp_hp]])

        ## receive action from gym
        # print("waitting for step in {}".format(self.pipe))
        request = self.pipe.recv()
        # print("get step in {}".format(self.pipe))
        if len(request) == 2 and request[0] == "step":
            action = request[1]
            command = self.action_strs[action]
            if command == "NEUTRAL":
                self.frozen_frames = 6
            else:
                self.frozen_frames = 0
                if command == "CROUCH_GUARD":
                    command = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
                elif command == "STAND_GUARD":
                    command = "4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4"
                self.cc.commandCall(command)
                self.inputKey = self.cc.getSkillKey()

    def get_reward(self, player):
        try:
            if self.pre_framedata.getEmptyFlag() or self.frameData.getEmptyFlag():
                reward = 0
            else:
                p2_hp_pre = self.pre_framedata.getCharacter(False).getHp()
                p1_hp_pre = self.pre_framedata.getCharacter(True).getHp()
                p2_hp_now = self.frameData.getCharacter(False).getHp()
                p1_hp_now = self.frameData.getCharacter(True).getHp()
                x_dist_pre = self.pre_framedata.getDistanceX()
                x_dist_now = self.frameData.getDistanceX()
                if player:
                    reward = ((p2_hp_pre - p2_hp_now) - (p1_hp_pre - p1_hp_now)) / 10
                else:
                    reward = ((p1_hp_pre - p1_hp_now) - (p2_hp_pre - p2_hp_now)) / 10
                if x_dist_now < x_dist_pre:
                    bonus = +0.01
                elif x_dist_now > x_dist_pre:
                    bonus = -0.01
                else:
                    bonus = 0
                reward += bonus
        except:
            reward = 0
        return reward

    def get_obs(self, frame_data, player, clip=True):
        my = frame_data.getCharacter(player)
        opp = frame_data.getCharacter(not player)

        # my information
        myEnergy = my.getEnergy() / 300
        myX = ((my.getLeft() + my.getRight()) / 2 - 960 / 2) / (960 / 2)
        myY = ((my.getBottom() + my.getTop()) / 2) / 640
        mySpeedX = my.getSpeedX() / 20
        mySpeedY = my.getSpeedY() / 28
        myState = my.getState().ordinal()
        myAction = my.getAction().ordinal()
        myRemainingFrame = my.getRemainingFrame() / 70

        # opp information
        oppEnergy = opp.getEnergy() / 300
        oppX = (
            (opp.getLeft() + opp.getRight()) / 2 - (my.getLeft() + my.getRight()) / 2
        ) / 960
        oppY = ((opp.getBottom() + opp.getTop()) / 2) / 640
        oppSpeedX = opp.getSpeedX() / 20
        oppSpeedY = opp.getSpeedY() / 28
        oppState = opp.getState().ordinal()
        oppAction = opp.getAction().ordinal()
        oppRemainingFrame = opp.getRemainingFrame() / 70

        observation = []
        
        # Add remainting time obs, because remaining time is crutial for the v net to estimate the value of the state
        # observation.append(frame_data.getRemainingTime() / 60.0)

        # my information
        observation.append(myEnergy)  # [0,1]
        observation.append(myX)  # [-1,1]
        observation.append(myY)  # [0,1]
        observation.append(mySpeedX)  # [-1,1]
        observation.append(mySpeedY)  # [-1,1]
        for i in range(4):
            if i == myState:  # [0,1]
                observation.append(1)
            else:
                observation.append(0)
        for i in range(56):
            if i == myAction:  # [0,1]
                observation.append(1)
            else:
                observation.append(0)
        observation.append(myRemainingFrame)  # [0,1]

        # opp information
        observation.append(oppEnergy)  # [0,1]
        observation.append(oppX)  # [-1,1]
        observation.append(oppY)  # [0,1]
        observation.append(oppSpeedX)  # [-1,1]
        observation.append(oppSpeedY)  # [-1,1]
        for i in range(4):
            if i == oppState:  # [0,1]
                observation.append(1)
            else:
                observation.append(0)
        for i in range(56):
            if i == oppAction:  # [0,1]
                observation.append(1)
            else:
                observation.append(0)
        observation.append(oppRemainingFrame)  # [0,1]

        if player:
            myProjectiles = self.frameData.getProjectilesByP1()
            oppProjectiles = self.frameData.getProjectilesByP2()
        else:
            myProjectiles = self.frameData.getProjectilesByP2()
            oppProjectiles = self.frameData.getProjectilesByP1()

        if len(myProjectiles) == 2:
            myHitDamage = myProjectiles[0].getHitDamage() / 200.0
            myHitAreaNowX = (
                (
                    myProjectiles[0].getCurrentHitArea().getLeft()
                    + myProjectiles[0].getCurrentHitArea().getRight()
                )
                / 2
                - (my.getLeft() + my.getRight()) / 2
            ) / 960.0
            myHitAreaNowY = (
                (
                    myProjectiles[0].getCurrentHitArea().getTop()
                    + myProjectiles[0].getCurrentHitArea().getBottom()
                )
                / 2
            ) / 640.0
            observation.append(myHitDamage)  # [0,1]
            observation.append(myHitAreaNowX)  # [-1,1]
            observation.append(myHitAreaNowY)  # [0,1]
            myHitDamage = myProjectiles[1].getHitDamage() / 200.0
            myHitAreaNowX = (
                (
                    myProjectiles[1].getCurrentHitArea().getLeft()
                    + myProjectiles[1].getCurrentHitArea().getRight()
                )
                / 2
                - (my.getLeft() + my.getRight()) / 2
            ) / 960.0
            myHitAreaNowY = (
                (
                    myProjectiles[1].getCurrentHitArea().getTop()
                    + myProjectiles[1].getCurrentHitArea().getBottom()
                )
                / 2
            ) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
        elif len(myProjectiles) == 1:
            myHitDamage = myProjectiles[0].getHitDamage() / 200.0
            myHitAreaNowX = (
                (
                    myProjectiles[0].getCurrentHitArea().getLeft()
                    + myProjectiles[0].getCurrentHitArea().getRight()
                )
                / 2
                - (my.getLeft() + my.getRight()) / 2
            ) / 960.0
            myHitAreaNowY = (
                (
                    myProjectiles[0].getCurrentHitArea().getTop()
                    + myProjectiles[0].getCurrentHitArea().getBottom()
                )
                / 2
            ) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
            for t in range(3):
                observation.append(0.0)
        else:
            for t in range(6):
                observation.append(0.0)

        if len(oppProjectiles) == 2:
            oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
            oppHitAreaNowX = (
                (
                    oppProjectiles[0].getCurrentHitArea().getLeft()
                    + oppProjectiles[0].getCurrentHitArea().getRight()
                )
                / 2
                - (my.getLeft() + my.getRight()) / 2
            ) / 960.0
            oppHitAreaNowY = (
                (
                    oppProjectiles[0].getCurrentHitArea().getTop()
                    + oppProjectiles[0].getCurrentHitArea().getBottom()
                )
                / 2
            ) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
            oppHitDamage = oppProjectiles[1].getHitDamage() / 200.0
            oppHitAreaNowX = (
                (
                    oppProjectiles[1].getCurrentHitArea().getLeft()
                    + oppProjectiles[1].getCurrentHitArea().getRight()
                )
                / 2
                - (my.getLeft() + my.getRight()) / 2
            ) / 960.0
            oppHitAreaNowY = (
                (
                    oppProjectiles[1].getCurrentHitArea().getTop()
                    + oppProjectiles[1].getCurrentHitArea().getBottom()
                )
                / 2
            ) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
        elif len(oppProjectiles) == 1:
            oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
            oppHitAreaNowX = (
                (
                    oppProjectiles[0].getCurrentHitArea().getLeft()
                    + oppProjectiles[0].getCurrentHitArea().getRight()
                )
                / 2
                - (my.getLeft() + my.getRight()) / 2
            ) / 960.0
            oppHitAreaNowY = (
                (
                    oppProjectiles[0].getCurrentHitArea().getTop()
                    + oppProjectiles[0].getCurrentHitArea().getBottom()
                )
                / 2
            ) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
            for t in range(3):
                observation.append(0.0)
        else:
            for t in range(6):
                observation.append(0.0)

        observation = np.array(observation, dtype=np.float32)
        if clip:
            observation = np.clip(observation, -1, 1)
        return observation

    # BEGIN MY CODE
    def get_representation(self, frame_data, player):
        """
        Get the representation of the state (at frame_data, player (True for player1, False for player2) is about to act)
        """
        obs = self.get_obs(frame_data, player, False)
        obs = np.append(obs, frame_data.getRemainingFramesNumber())
        return tuple(obs)

    def check_game_result(self, frame_data, player, time_limit=60):
        if time_limit > 60:
            raise ValueError("time_limit should be less than 60")
        if frame_data.getEmptyFlag():
            return 0
        if frame_data.getCharacter(player).getHp() <= 0:
            return -1
        if frame_data.getCharacter(not player).getHp() <= 0:
            return 1
        if frame_data.getRemainingTime() <= 60 - time_limit:
            print("Simulate to the end of the game:")
            print("P1 HP: ", frame_data.getCharacter(True).getHp())
            print("P2 HP: ", frame_data.getCharacter(False).getHp())
            if (
                frame_data.getCharacter(player).getHp()
                > frame_data.getCharacter(not player).getHp()
            ):
                return 1
            # When game ended with draw, return -1
            return -1
        return 0
    
    def map_action(self, action):
        command = self.action_strs[action]
        if command == "AIR_A":
            return self.gateway.jvm.enumerate.Action.AIR_A
        elif command == "AIR_B":
            return self.gateway.jvm.enumerate.Action.AIR_B
        elif command == "AIR_D_DB_BA":
            return self.gateway.jvm.enumerate.Action.AIR_D_DB_BA
        elif command == "AIR_D_DB_BB":
            return self.gateway.jvm.enumerate.Action.AIR_D_DB_BB
        elif command == "AIR_D_DF_FA":
            return self.gateway.jvm.enumerate.Action.AIR_D_DF_FA
        elif command == "AIR_D_DF_FB":
            return self.gateway.jvm.enumerate.Action.AIR_D_DF_FB
        elif command == "AIR_DA":
            return self.gateway.jvm.enumerate.Action.AIR_DA
        elif command == "AIR_DB":
            return self.gateway.jvm.enumerate.Action.AIR_DB
        elif command == "AIR_F_D_DFA":
            return self.gateway.jvm.enumerate.Action.AIR_F_D_DFA
        elif command == "AIR_F_D_DFB":
            return self.gateway.jvm.enumerate.Action.AIR_F_D_DFB
        elif command == "AIR_FA":
            return self.gateway.jvm.enumerate.Action.AIR_FA
        elif command == "AIR_FB":
            return self.gateway.jvm.enumerate.Action.AIR_FB
        elif command == "AIR_UA":
            return self.gateway.jvm.enumerate.Action.AIR_UA
        elif command == "AIR_UB":
            return self.gateway.jvm.enumerate.Action.AIR_UB
        elif command == "BACK_JUMP":
            return self.gateway.jvm.enumerate.Action.BACK_JUMP
        elif command == "BACK_STEP":
            return self.gateway.jvm.enumerate.Action.BACK_STEP
        elif command == "CROUCH_A":
            return self.gateway.jvm.enumerate.Action.CROUCH_A
        elif command == "CROUCH_B":
            return self.gateway.jvm.enumerate.Action.CROUCH_B
        elif command == "CROUCH_FA":
            return self.gateway.jvm.enumerate.Action.CROUCH_FA
        elif command == "CROUCH_FB":
            return self.gateway.jvm.enumerate.Action.CROUCH_FB
        elif command == "CROUCH_GUARD":
            return self.gateway.jvm.enumerate.Action.CROUCH_GUARD
        elif command == "DASH":
            return self.gateway.jvm.enumerate.Action.DASH
        elif command == "FOR_JUMP":
            return self.gateway.jvm.enumerate.Action.FOR_JUMP
        elif command == "FORWARD_WALK":
            return self.gateway.jvm.enumerate.Action.FORWARD_WALK
        elif command == "JUMP":
            return self.gateway.jvm.enumerate.Action.JUMP
        elif command == "NEUTRAL":
            return self.gateway.jvm.enumerate.Action.NEUTRAL
        elif command == "STAND_A":
            return self.gateway.jvm.enumerate.Action.STAND_A
        elif command == "STAND_B":
            return self.gateway.jvm.enumerate.Action.STAND_B
        elif command == "STAND_D_DB_BA":
            return self.gateway.jvm.enumerate.Action.STAND_D_DB_BA
        elif command == "STAND_D_DB_BB":
            return self.gateway.jvm.enumerate.Action.STAND_D_DB_BB
        elif command == "STAND_D_DF_FA":
            return self.gateway.jvm.enumerate.Action.STAND_D_DF_FA
        elif command == "STAND_D_DF_FB":
            return self.gateway.jvm.enumerate.Action.STAND_D_DF_FB
        elif command == "STAND_D_DF_FC":
            return self.gateway.jvm.enumerate.Action.STAND_D_DF_FC
        elif command == "STAND_F_D_DFA":
            return self.gateway.jvm.enumerate.Action.STAND_F_D_DFA
        elif command == "STAND_F_D_DFB":
            return self.gateway.jvm.enumerate.Action.STAND_F_D_DFB
        elif command == "STAND_FA":
            return self.gateway.jvm.enumerate.Action.STAND_FA
        elif command == "STAND_FB":
            return self.gateway.jvm.enumerate.Action.STAND_FB
        elif command == "STAND_GUARD":
            return self.gateway.jvm.enumerate.Action.STAND_GUARD
        elif command == "THROW_A":
            return self.gateway.jvm.enumerate.Action.THROW_A
        elif command == "THROW_B":
            return self.gateway.jvm.enumerate.Action.THROW_B
    
    def get_action(self, action):
        if action is None:
            return self.gateway.jvm.java.util.ArrayDeque()
        command = self.map_action(action)
        actions = self.gateway.jvm.java.util.ArrayDeque()
        actions.add(command)
        return actions

    def simulate(self, frameData, player, action1, action2, frame_number):
        return self.simulator.simulate(frameData, player, self.get_action(action1), self.get_action(action2), frame_number)
    # END MY CODE

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
