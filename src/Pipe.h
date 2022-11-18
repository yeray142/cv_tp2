#pragma once

#include <SFML/Graphics.hpp>
#include "Game.h"
#include <vector>

class Pipe
{
public:
	Pipe(GameDataRef data);

	void SpawnBottomPipe();
	void SpawnTopPipe();
	void SpawnInvisiblePipe();
	void MovePipes(float dt);
	void DrawPipes();

private:
	GameDataRef _data;
	std::vector<sf::Sprite> pipeSprites;

};
