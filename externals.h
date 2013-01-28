#define _SCL_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#define GLM_FORCE_SSE2

#include <stdio.h>
#include <GL/glew.h>
#include <GL/wglew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include <mmsystem.h>
#include <stdlib.h>
#include <intrin.h>

#include <tbb/tbb.h>
#include <iostream>

#define POCO_STATIC
#include "Poco/Path.h"
#include "Poco/File.h"
#include "Poco/FileStream.h"
#include "Poco/Timestamp.h"
#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/ServerSocket.h"

#include "ist/ist.h"
#include "ist/Graphics.h"
#include "ist/Sound.h"
#include "features.h"
#include "types.h"
