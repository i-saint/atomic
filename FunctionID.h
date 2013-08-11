﻿#ifndef atm_FunctionID_h
#define atm_FunctionID_h

namespace atm {

istSEnumBlock(FunctionID,
    istSEnum(FID_unknown),

    istSEnum(FID_addDebugMenu),
    istSEnum(FID_kill),
    istSEnum(FID_destroy),
    istSEnum(FID_setRefCount),
    istSEnum(FID_incRefCount),
    istSEnum(FID_decRefCount),
    istSEnum(FID_damage),
    istSEnum(FID_eventCollide),
    istSEnum(FID_eventFluid),
    istSEnum(FID_eventDamage),
    istSEnum(FID_eventDestroy),
    istSEnum(FID_eventKill),

    istSEnum(FID_getRefCount),
    istSEnum(FID_getDiffuseColor),
    istSEnum(FID_setDiffuseColor),
    istSEnum(FID_getGlowColor),
    istSEnum(FID_setGlowColor),
    istSEnum(FID_getModel),
    istSEnum(FID_setModel),
    istSEnum(FID_getCollisionHandle),
    istSEnum(FID_getCollisionFlags),
    istSEnum(FID_setCollisionFlags),
    istSEnum(FID_getCollisionGroup),
    istSEnum(FID_setCollisionGroup),
    istSEnum(FID_setCollisionShape),
    istSEnum(FID_isDead),
    istSEnum(FID_getLife),
    istSEnum(FID_setLife),
    istSEnum(FID_getPastTime),
    istSEnum(FID_setPastTime),
    istSEnum(FID_getPower),
    istSEnum(FID_setPower),
    istSEnum(FID_getOwner),
    istSEnum(FID_setOwner),
    istSEnum(FID_getVelocity),
    istSEnum(FID_setVelocity),
    istSEnum(FID_getPivot),
    istSEnum(FID_setPivot),
    istSEnum(FID_getPosition),
    istSEnum(FID_setPosition),
    istSEnum(FID_move),
    istSEnum(FID_orient),
    istSEnum(FID_instruct),
    istSEnum(FID_getScale),
    istSEnum(FID_setScale),
    istSEnum(FID_getAxis),
    istSEnum(FID_setAxis),
    istSEnumEq(FID_getAxis1, FID_getAxis),
    istSEnumEq(FID_setAxis1, FID_setAxis),
    istSEnum(FID_getAxis2),
    istSEnum(FID_setAxis2),
    istSEnum(FID_getRotate),
    istSEnum(FID_setRotate),
    istSEnumEq(FID_getRotate1, FID_getRotate),
    istSEnumEq(FID_setRotate1, FID_setRotate),
    istSEnum(FID_getRotate2),
    istSEnum(FID_setRotate2),
    istSEnum(FID_getDirection),
    istSEnum(FID_setDirection),
    istSEnum(FID_getUpVector),
    istSEnum(FID_setUpVector),
    istSEnum(FID_getSpeed),
    istSEnum(FID_setSpeed),
    istSEnum(FID_getAccel),
    istSEnum(FID_setAccel),
    istSEnum(FID_getRotateSpeed),
    istSEnum(FID_setRotateSpeed),
    istSEnumEq(FID_getRotateSpeed1, FID_getRotateSpeed),
    istSEnumEq(FID_setRotateSpeed1, FID_setRotateSpeed),
    istSEnum(FID_getRotateSpeed2),
    istSEnum(FID_setRotateSpeed2),

    istSEnum(FID_pulse),
    istSEnum(FID_addSpinSpeed),
    istSEnum(FID_getSpinAngle),
    istSEnum(FID_setSpinAngle),
    istSEnum(FID_getSpinMinAngle),
    istSEnum(FID_setSpinMinAngle),
    istSEnum(FID_getSpinMaxAngle),
    istSEnum(FID_setSpinMaxAngle),
    istSEnum(FID_getSpinSpeed),
    istSEnum(FID_setSpinSpeed),
    istSEnum(FID_getSpinReturnSpeed),
    istSEnum(FID_setSpinReturnSpeed),
    istSEnum(FID_getSpinMaxSpeed),
    istSEnum(FID_setSpinMaxSpeed),
    istSEnum(FID_getSpinResist),
    istSEnum(FID_setSpinResist),
    istSEnum(FID_getSpinDecel),
    istSEnum(FID_setSpinDecel),
    istSEnum(FID_getSpinOneWay),
    istSEnum(FID_setSpinOneWay),

    istSEnum(FID_getParent),
    istSEnum(FID_setParent),
    istSEnum(FID_getTransformMatrix),
    istSEnum(FID_setTransformMatrix),
    istSEnum(FID_computeTransformMatrix),
    istSEnum(FID_computeRotationMatrix),
    istSEnum(FID_getInvTransformMatrix),
    istSEnum(FID_updateTransformMatrix),
    istSEnum(FID_setRoutine),
    istSEnum(FID_setLightRadius),
    istSEnum(FID_setExplosionSE),
    istSEnum(FID_setExplosionChannel),
    istSEnum(FID_setWeapon),
    istSEnum(FID_addBloodstain),
    istSEnum(FID_getRadius),
    istSEnum(FID_setRadius),
    istSEnum(FID_getColor),
    istSEnum(FID_setColor),
    istSEnum(FID_getDiffuse),
    istSEnum(FID_setDiffuse),
    istSEnum(FID_getAmbient),
    istSEnum(FID_setAmbient),
    istSEnum(FID_addPositionXCP),
    istSEnum(FID_setPositionXCP),
    istSEnum(FID_erasePositionXCP),
    istSEnum(FID_clearPositionXCP),
    istSEnum(FID_addPositionYCP),
    istSEnum(FID_setPositionYCP),
    istSEnum(FID_erasePositionYCP),
    istSEnum(FID_clearPositionYCP),
    istSEnum(FID_addRotationCP),
    istSEnum(FID_setRotationCP),
    istSEnum(FID_eraseRotationCP),
    istSEnum(FID_clearRotationCP),

    istSEnum(FID_getPositionAbs),
    istSEnum(FID_setPositionAbs),
    istSEnum(FID_getDirectionAbs),
    istSEnum(FID_setDirectionAbs),
    istSEnum(FID_setLifeTime),
    istSEnum(FID_getLifeTime),
    istSEnum(FID_setGear),
    istSEnum(FID_setBlock),
    istSEnum(FID_setSlideDir),
    istSEnum(FID_setLinkSpeed),
    istSEnum(FID_poke),

    istSEnum(FID_End),
);

} // namespace atm

#endif atm_FunctionID_h
