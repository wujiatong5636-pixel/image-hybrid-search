"""类别兼容规则测试。"""

from src.image_classifier import categories_compatible


def test_living_room_accepts_furniture_but_rejects_kitchen():
    assert categories_compatible("living_room", "living_room")
    assert categories_compatible("living_room", "furniture")
    assert not categories_compatible("living_room", "kitchen_sink")


def test_sink_accepts_kitchen_cabinet_but_rejects_bathroom():
    assert categories_compatible("kitchen_sink", "kitchen_cabinet")
    assert not categories_compatible("kitchen_sink", "bathroom")


def test_cleaning_is_strict():
    assert categories_compatible("cleaning", "cleaning")
    assert not categories_compatible("cleaning", "construction")
