// 太阳/月亮在天空中的视角直径（单位度）;
const float sunAngularSize = 0.533333;
const float moonAngularSize = 0.516667;

// 天空系数与高度

#define airNumberDensity 2.5035422e25	// 空气分子数密度
#define ozoneConcentrationPeak 8e-6		// 臭氧峰值体积浓度（8 ppm）
const float ozoneNumberDensity = airNumberDensity * ozoneConcentrationPeak;		// 臭氧数密度
#define ozoneCrossSection vec3(4.51103766177301e-21, 3.2854797958699e-21, 1.96774621921165e-22)	//单分子臭氧吸收截面（针对RGB三段波） 

#define sky_planetRadius 6731e3		// 地球半径（米）
#define sky_atmosphereHeight 110e3	// 大气层高度 110 km
#define sky_scaleHeights vec2(8.0e3, 1.2e3)   // 瑞利高标 8km，米氏高标 1.2km

// 瑞利/米氏散射系数，GB通道比R通道更大，因为蓝光散射更强。具体数值由外部宏（sky_config.glsl）提供
#define sky_coefficientRayleigh vec3(sky_coefficientRayleighR*1e-6, sky_coefficientRayleighG*1e-5, sky_coefficientRayleighB*1e-5)
#define sky_coefficientMie vec3(sky_coefficientMieR*1e-6, sky_coefficientMieG*1e-6, sky_coefficientMieB*1e-6) // 经验下的米氏散射系数下限 >= 2e-6

const vec3 sky_coefficientOzone = (ozoneCrossSection * (ozoneNumberDensity * 0.2e-6)); // 臭氧吸收系数（单分子臭氧吸收截面*臭氧数密度*针对于该情景下的修正因子0.2e-6）


// 将物理参数转换为渲染优化格式
const vec2 sky_inverseScaleHeights = 1.0 / sky_scaleHeights;    // 大气密度积分公式中会用到标高倒数，由于GPU处理除法比乘法慢得多，所以预先得到标高倒数
const vec2 sky_scaledPlanetRadius = sky_planetRadius * sky_inverseScaleHeights;  // 行星半径归一化！把参与运算的所有数值拉到同一个安全的量级范围内，确保浮点数在整个大气层（从地面到太空边缘）都能保持足够的有效位数
const float sky_atmosphereRadius = sky_planetRadius + sky_atmosphereHeight;    // 大气层顶半径（行星半径 + 大气厚度）。用于射线-球体相交测试，判断光线是否还在大气层内。
const float sky_atmosphereRadiusSquared = sky_atmosphereRadius * sky_atmosphereRadius;    // 预计算平方值。射线与球体的求交公式要用

// 散射系数的矩阵打包，2*3的矩阵，第一列瑞利散射系数（RGB），第二列米氏（RGB），臭氧不参与散射，只负责衰减吸收
#define sky_coefficientsScattering mat2x3(sky_coefficientRayleigh, sky_coefficientMie)

// 消光矩阵，用于计算光学深度
const mat3 sky_coefficientsAttenuation = mat3(sky_coefficientRayleigh , sky_coefficientMie, sky_coefficientOzone ); 



// 相位函数（描述光散射的方向性）

// 瑞利相位函数（简化线性版本）
float sky_rayleighPhase(float cosTheta) 
{
	const vec2 mul_add = vec2(0.1, 0.28) * rPI;  // 乘加运算包含两个拟合系数，GPU通过一条FMA指令完成整个瑞利相位函数的运算
	return cosTheta * mul_add.x + mul_add.y; // optimized version from [Elek09], divided by 4 pi for energy conservation
}

// 米氏相位函数
float sky_miePhase(float cosTheta, const float g) 
{
	float gg = g * g;  // g（各向异性因子） > 0 表示强前向散射，典型值：晴朗天空 g≈0.76 ，雾霾天 g≈0.9
	return (gg * -0.25 + 0.25) * rPI * pow(-(2.0 * g) * cosTheta + (gg + 1.0), -1.5);
}

// 将相位函数打包成vec2双通道数据容器，高效利用GPU的向量并行能力
vec2 sky_phase(float cosTheta, const float g) 
{
	return vec2(sky_rayleighPhase(cosTheta), sky_miePhase(cosTheta, g));
}

// 大气密度分布 
// centerDistance：当前采样点到地心的距离
// 大气密度随高度h呈指数衰减，ρ(h)=ρ0*e^(-h/H)，高度 h = centerDistance - sky_planetRadius
vec3 sky_density(float centerDistance) 
{
	vec2 rayleighMie = exp(centerDistance * -sky_inverseScaleHeights + sky_scaledPlanetRadius);

	// 臭氧密度采用了基于 Sergeant Sarcasm 设计的分段指数模型 - https://www.desmos.com/calculator/j0wozszdwa
	// 该模型核心思想是臭氧浓度在35km高度达到峰值，并向两侧呈现不同速率的衰减。
	float ozone = exp(-max(0.0, (35000.0 - centerDistance) - sky_planetRadius) * (1.0 / 5000.0))
	            * exp(-max(0.0, (centerDistance - 35000.0) - sky_planetRadius) * (1.0 / 15000.0));
	return vec3(rayleighMie, ozone);	// rayleighMie.x存瑞利散射（空气分子）的密度；rayleighMie.y存米氏散射（气溶胶雾霾）的密度。
}


// 光线步进（Ray Marching）与数值积分算法：τ = ∫ρ(s)ds
vec3 sky_airmass(vec3 position, vec3 direction, float rayLength, const float steps) 
{
	// rayLength（相机沿着视线到大气边缘的距离）由重载提供
	float stepSize  = rayLength * (1.0 / steps);	// 步长
	vec3  increment = direction * stepSize;			// 步进向量，每次循环向前移动
	position += increment * 0.5;		// 半步偏移，对应数值积分中的中点法则：将起始位置向前推移半个步长，取整段的平均密度，效果更加平滑

	vec3 airmass = vec3(0.0);	// 初始化为0的三维向量，用来存储瑞利米氏臭氧的总密度累加值

	for (int i = 0; i < steps; ++i, position += increment) 
	{
		airmass += sky_density(length(position));	// 路径积分近似为离散求和
	}

	return airmass * stepSize;	//	最后一定乘上步长，也就是ds，得到光学深度
}

// 重载函数，根据射线与球体求交算法，得到光线总长 rayLength
vec3 sky_airmass(vec3 position, vec3 direction, const float steps) 
{
	float rayLength = dot(position, direction);   // 对应一元二次方程的 b/2
	      rayLength = rayLength * rayLength + sky_atmosphereRadiusSquared - dot(position, position);   // 这里求判别式把4约掉了!
		  if (rayLength < 0.0) 
			  return vec3(0.0);
		  rayLength = sqrt(rayLength) - dot(position, direction);   // 得到正根，即raylenth

	return sky_airmass(position, direction, rayLength, steps);
}

// 由密度计算光学深度（天空颜色的核心变量）
// 光线穿过介质时的衰减遵循比尔-朗伯定律，透光度由光线的光学深度决定
// 这里涉及矩阵的3*3乘法，累积密度*消光矩阵
vec3 sky_opticalDepth(vec3 position, vec3 direction, float rayLength, const float steps) 
{
	return sky_coefficientsAttenuation * sky_airmass(position, direction, rayLength, steps);
}

vec3 sky_opticalDepth(vec3 position, vec3 direction, const float steps) 
{
	return sky_coefficientsAttenuation * sky_airmass(position, direction, steps);
}

// 透光率
vec3 sky_transmittance(vec3 position, vec3 direction, const float steps) 
{
	return exp(-sky_opticalDepth(position, direction, steps) * rLOG2);
	// rLOG2 = log2(e)，改性能优化在于GPU处理以2为底的幂运算极快，上面代码在着色器编译器里会转变为一条极其廉价的指令exp2(...)
}




vec3 calculateAtmosphere(vec3 background, vec3 viewVector, vec3 upVector, vec3 sunVector, vec3 moonVector, out vec2 pid, out vec3 transmittance, const int iSteps, float noise) 
{
	const int jSteps = 4;

	// 计算地面暗化因子
	#ifdef SKY_GROUND
		#if CUMULONIMBUS > 1	// 若启动了积雨云，参考高度6000m
			const float referenceHeight = 6000.0;
		#else
			const float referenceHeight = CloudLayer0_height;	// 否则基础云层高度
		#endif
		float heightRelativeToClouds = clamp(1.0 - max(cameraPosition.y - referenceHeight,0.0) / 100.0 ,0.0,1.0);
		float planetGround = mix(exp(-1.0 * pow(max(-viewVector.y*10. + 0.1,0.0),2.)), exp(-100.0 * pow(max(-viewVector.y*5. + 0.1,0.0),2.)), heightRelativeToClouds); // darken the ground in the sky.
	#else
		float planetGround = pow(clamp(viewVector.y+1.0,0.0,1.0),2); // darken the ground in the sky.
	#endif
	
	float GroundDarkening = max(planetGround * 0.7+0.3,clamp(sunVector.y*2.0,0.0,1.0));

	// 相机位置
	vec3 viewPos = (sky_planetRadius + 1.0 + max(eyeAltitude-300.0,0.0)) * upVector;	// 计算观察点位置，若超过300m，则观测点在竖直方向加上超出部分

	// 视线与大气求交
	vec2 aid = rsi(viewPos, viewVector, sky_atmosphereRadius);
	if (aid.y < 0.0) {transmittance = vec3(1.0); return vec3(0.0);}	// 条件表明若视线根本没触碰大气层（如在太空望向深空），透光率设定为1（完全不衰减），并返回黑色

	// 视线与地表求交
	pid = rsi(viewPos, viewVector, sky_planetRadius * 0.998);
	bool planetIntersected = pid.y > 0.0;

	// 确定真实的步进空间，sd.x是步进的起点距离（光线进入大气层的有效起点距离），sd.y是步进的终点距离。这段代码简单来说就是在大气层内，sd.x=0，在太空中，sd.x视线先穿过的真空距离
	vec2 sd = vec2((planetIntersected && pid.x < 0.0) ? pid.y : max(aid.x, 0.0), (planetIntersected && pid.x > 0.0) ? pid.x : aid.y);

	// 计算步长与步进向量。(sd.y - sd.x)就是光线在大气中的真实物理长度
	float stepSize  = (sd.y - sd.x) * (1.0 / float(iSteps));
	vec3  increment = viewVector * stepSize;    // 每次循环位置需要移动的向量

	// 添加随机噪声进行抖动防锯齿
	vec3  position  = viewVector * sd.x + viewPos;	// 移步至采样点
	position += increment * (0.34*noise);

	// 光学预处理步骤，计算天体散射相函数
	vec2 phaseSun = sky_phase(dot(viewVector, sunVector), 0.8);
	vec2 phaseMoon = sky_phase(dot(viewVector, moonVector), 0.8);

	// 日食
//	#ifdef CUSTOM_MOON_ROTATION
//		float eclipseDarkeness = smoothstep(0.005, 0.08, length(sunVector-moonVector));
//		phaseSun *= mix(1.0, eclipseDarkeness, smoothstep(-1.0, 0.175, viewVector.y));
//	#endif

	// 光线步进循环前的初始化与预处理
	vec3 scatteringSun     = vec3(0.0);
	vec3 scatteringMoon    = vec3(0.0);
	vec3 scatteringAmbient = vec3(0.0);

	transmittance = vec3(1.0);

	// 日出日落的颜色欺骗，sunVector.y：太阳高度
	float high_sun = clamp(pow(sunVector.y+0.6,5.),0.0,1.0) * 3.0; // make sunrise less blue, and allow sunset to be bluer
	float low_sun = clamp(((1.0-abs(sunVector.y))*3.) - high_sun,1.0,2.0) ;
	

	// 光线步进循环核心
	for (int i = 0; i < iSteps; ++i, position += increment) {
		vec3 density = sky_density(length(position));
		if (density.y > 1e35) break;	// 提前终止优化：若米氏密度返回该值，说明求密度函数检测的光线已经撞击到了地面或者飞出了大气层顶
		vec3 stepAirmass      = density * stepSize ;	// 步进气质量
		vec3 stepOpticalDepth = sky_coefficientsAttenuation * stepAirmass;	// 步进光学深度

		// 利用解析积分求单个步长内的散射
		vec3 stepTransmittance       = exp2(-stepOpticalDepth * rLOG2);
		vec3 stepTransmittedFraction = clamp01((stepTransmittance - 1.0) / -stepOpticalDepth) ;		// 对微分方程进行了精确解析解（硬件级优化）-->极其考验数学与GPU架构！
		vec3 stepScatteringVisible   = transmittance * stepTransmittedFraction ;    // 将“步内逃逸比例”乘以“从相机到这一步起点的累计透射率，得到这一步的散射光最终能到达相机的真实比例
		
		// 计算到达相机的太阳光和月光
		scatteringSun  += sky_coefficientsScattering  * (stepAirmass.xy * phaseSun) * stepScatteringVisible * sky_transmittance(position, sunVector, jSteps) * planetGround;
		scatteringMoon += sky_coefficientsScattering * (stepAirmass.xy * phaseMoon) * stepScatteringVisible * sky_transmittance(position, moonVector, jSteps);
		
		// 模拟的多重散射
		scatteringAmbient += sky_coefficientsScattering * stepAirmass.xy * stepScatteringVisible * low_sun;
		
		// 更新累计透射率
		transmittance *= stepTransmittance;
	}
	
	vec3 scattering = scatteringAmbient * background + scatteringSun * sunColorBase + scatteringMoon*moonColorBase * 0.5;

	return scattering;
}
