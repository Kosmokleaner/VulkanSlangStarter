#include "global.h"	// include first in any cpp

#include "GlmMathTests.h"
#include "TestFramework.h"
#include "Transform.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace tests
{
	static const char* kSuiteMatrix = "GlmMatrix";
	static const char* kSuiteQuat = "GlmQuaternion";
	static const char* kSuiteTransform = "EngineTransform";

	//---- GlmMatrix ------------------------------------------------------------------------
	// glm::mat4 is column-major in memory (m[col][row]) and multiplies as M * v with v a
	// column vector, i.e. for A * B * v, B is applied to v first, then A. This is the
	// opposite of DirectX's row-vector v * M1 * M2 convention where M1 applies first.

	static void Test_IdentityIsNoOp(TestReport& report)
	{
		glm::mat4 id(1.0f);
		float4 v(1.0f, 2.0f, 3.0f, 1.0f);
		float4 result = id * v;

		report.Check(kSuiteMatrix, "identity_is_noop", NearlyEqual(result, v),
			"mat4(1.0f) * v must return v unchanged, got " + ToString(float3(result)));
	}

	static void Test_TranslationMovesPointsNotDirections(TestReport& report)
	{
		float3 t(5.0f, 0.0f, 0.0f);
		glm::mat4 T = glm::translate(glm::mat4(1.0f), t);

		// w=1 -> a point: translation applies
		float4 point = T * float4(1, 0, 0, 1);
		float3 expectedPoint(6, 0, 0);
		report.Check(kSuiteMatrix, "translation_moves_point", NearlyEqual(float3(point), expectedPoint),
			"translate(5,0,0) * point(1,0,0,w=1) should be " + ToString(expectedPoint) + ", got " + ToString(float3(point)));

		// w=0 -> a direction: translation must NOT apply, this is why direction vectors
		// (normals, ray directions, TransformPosition deltas) use w=0 / vec3 overloads
		float4 dir = T * float4(1, 0, 0, 0);
		float3 expectedDir(1, 0, 0);
		report.Check(kSuiteMatrix, "translation_ignores_direction", NearlyEqual(float3(dir), expectedDir),
			"translate(5,0,0) * direction(1,0,0,w=0) should stay " + ToString(expectedDir) + ", got " + ToString(float3(dir)));
	}

	static void Test_MatrixMultiplyOrderMatters(TestReport& report)
	{
		// Same convention Transform::GetObjectMatrix uses: translate(pos) * scale(s) * mat4(rot),
		// i.e. rotate first, then scale, then translate, when applied to an object-space point.
		float3 p(1, 0, 0);
		glm::mat4 T = glm::translate(glm::mat4(1.0f), float3(5, 0, 0));
		glm::mat4 S = glm::scale(glm::mat4(1.0f), float3(2, 1, 1));

		// T * S: S (rightmost) applies first -> p scaled to (2,0,0), then translated -> (7,0,0)
		float3 resultTS = float3((T * S) * float4(p, 1));
		float3 expectedTS(7, 0, 0);
		report.Check(kSuiteMatrix, "multiply_order_T_times_S", NearlyEqual(resultTS, expectedTS),
			"(T * S) * p : S applies first (scale), then T (translate) -> expected " + ToString(expectedTS) + ", got " + ToString(resultTS));

		// S * T: T (rightmost) applies first -> p translated to (6,0,0), then scaled -> (12,0,0)
		float3 resultST = float3((S * T) * float4(p, 1));
		float3 expectedST(12, 0, 0);
		report.Check(kSuiteMatrix, "multiply_order_S_times_T", NearlyEqual(resultST, expectedST),
			"(S * T) * p : T applies first (translate), then S (scale) -> expected " + ToString(expectedST) + ", got " + ToString(resultST)
			+ " -- proves matrix multiply is not commutative and order must match the intended transform sequence");
	}

	static void Test_MultiplyIsAssociative(TestReport& report)
	{
		glm::mat4 A = glm::translate(glm::mat4(1.0f), float3(1, 2, 3));
		glm::mat4 B = glm::scale(glm::mat4(1.0f), float3(2, 3, 4));
		float4 v(1, 1, 1, 1);

		float4 lhs = (A * B) * v;
		float4 rhs = A * (B * v);

		report.Check(kSuiteMatrix, "multiply_associative", NearlyEqual(lhs, rhs),
			"(A*B)*v must equal A*(B*v); got " + ToString(float3(lhs)) + " vs " + ToString(float3(rhs)));
	}

	static void Test_ColumnMajorStorage(TestReport& report)
	{
		// glm::mat4 is stored column-major: m[col][row]. A translation matrix's translation
		// component lives entirely in column 3 (m[3] = (tx,ty,tz,1)), not in row 3.
		// This is exactly the fact that forces this engine to glm::transpose() matrices
		// before uploading to shaders declared row_major (see App.cpp around
		// SceneInfo upload, and shaders/shader_io.h's ROW_MAJOR macro): the raw bytes are
		// column-major, so the shader side must be told to reinterpret rows as columns.
		float3 t(7, 8, 9);
		glm::mat4 T = glm::translate(glm::mat4(1.0f), t);

		float3 column3(T[3]);
		report.Check(kSuiteMatrix, "column_major_translation_in_column3", NearlyEqual(column3, t),
			"glm stores mat4 column-major, so translate()'s offset must appear in m[3] (the 4th column), expected "
			+ ToString(t) + ", got " + ToString(column3));

		float3 row3(T[0][3], T[1][3], T[2][3]);
		float3 expectedRow3(0, 0, 0);
		report.Check(kSuiteMatrix, "column_major_translation_not_in_row3", NearlyEqual(row3, expectedRow3),
			"row 3 (m[0][3], m[1][3], m[2][3]) must stay 0 for a pure translation, got " + ToString(row3));
	}

	static void Test_QuatMatrixAgree(TestReport& report)
	{
		// mat4(quat) must rotate identically to quat * vec3, since Transform::GetObjectMatrix
		// uses mat4(rot) while Transform::TransformPosition/GetForward use rot * v directly --
		// both must agree for the object matrix to be a faithful representation of `rot`.
		glm::quat q = glm::angleAxis(glm::radians(90.0f), float3(0, 1, 0));
		float3 v(0, 0, 1);

		float3 viaQuat = q * v;
		float3 viaMatrix = float3(glm::mat4(q) * float4(v, 0.0f));

		report.Check(kSuiteMatrix, "quat_and_mat4_rotate_the_same_way", NearlyEqual(viaQuat, viaMatrix),
			"q * v and mat4(q) * vec4(v,0) must agree, got " + ToString(viaQuat) + " vs " + ToString(viaMatrix));
	}

	//---- GlmQuaternion ---------------------------------------------------------------------
	// glm::quat's constructor order is (w, x, y, z) -- note this differs from float4's
	// (x, y, z, w), a common source of bugs when the two are confused.

	static void Test_QuatConstructorArgumentOrderIsWXYZ(TestReport& report)
	{
		// Transform's default constructor (Transform.h) writes glm::quat(1, 0, 0, 0) for
		// "no rotation". If the argument order were (x,y,z,w) instead, this would be a
		// 180 degree rotation around X, not identity -- so this test pins down which it is.
		glm::quat q(1, 0, 0, 0);
		float3 v(1, 2, 3);
		float3 rotated = q * v;

		report.Check(kSuiteQuat, "constructor_order_is_w_x_y_z", NearlyEqual(rotated, v),
			"glm::quat(1,0,0,0) must be the identity rotation (w=1,x=y=z=0), so q*v == v; expected "
			+ ToString(v) + ", got " + ToString(rotated) + " -- if this fails, glm's quat constructor order assumed by Transform.h's default constructor is wrong");
	}

	static void Test_QuatDoubleCover(TestReport& report)
	{
		// q and -q represent the same rotation (double cover of SO(3)); both must rotate
		// a vector identically even though the quaternion components differ in sign.
		glm::quat q = glm::angleAxis(glm::radians(37.0f), glm::normalize(float3(1, 1, 1)));
		glm::quat negQ(-q.w, -q.x, -q.y, -q.z);
		float3 v(1, 0, 0);

		report.Check(kSuiteQuat, "double_cover_q_and_negq_rotate_same", NearlyEqual(q * v, negQ * v),
			"q and -q must rotate a vector identically, got " + ToString(q * v) + " vs " + ToString(negQ * v));
	}

	static void Test_QuatCompositionOrder(TestReport& report)
	{
		// For quaternions p = a * b, applying p to a vector applies b first, then a:
		// (a * b) * v == a * (b * v). Same rule as matrices (Test_MatrixMultiplyOrderMatters
		// above), since glm's quat-vector rotation follows the same column-vector convention.
		glm::quat a = glm::angleAxis(glm::radians(90.0f), float3(0, 0, 1));
		glm::quat b = glm::angleAxis(glm::radians(90.0f), float3(1, 0, 0));
		float3 v(0, 1, 0);

		float3 composed = (a * b) * v;
		float3 sequential = a * (b * v);

		report.Check(kSuiteQuat, "compose_order_b_applied_first", NearlyEqual(composed, sequential),
			"(a*b)*v must equal a*(b*v) (b applied first, then a); got " + ToString(composed) + " vs " + ToString(sequential));
	}

	//---- EngineTransform -------------------------------------------------------------------
	// Cross-checks Transform.h's own composition (operator*, TransformPosition) against the
	// raw glm behavior documented above, since Transform.h wraps rot/position but does not
	// change the underlying convention.

	static void Test_TransformCompositionMatchesSequentialApplication(TestReport& report)
	{
		Transform a;
		a.SetRotation(float2(glm::radians(90.0f), 0.0f)); // yaw 90
		a.SetPos(float3(10, 0, 0));

		Transform b;
		b.SetRotation(float2(0.0f, glm::radians(30.0f))); // pitch 30
		b.SetPos(float3(0, 1, 0));

		Transform c = a * b;

		float3 x(1, 2, 3);

		// Transform.h's own comment on operator* claims "a followed by b"; this test checks
		// the actual observed behavior against both possible readings so the ambiguous
		// comment can be resolved from real output instead of guesswork.
		float3 viaComposed = c.TransformPosition(x);
		float3 bThenA = a.TransformPosition(b.TransformPosition(x));  // b's local space nested inside a's
		float3 aThenB = b.TransformPosition(a.TransformPosition(x));  // a's local space nested inside b's

		bool matchesBThenA = NearlyEqual(viaComposed, bThenA);
		bool matchesAThenB = NearlyEqual(viaComposed, aThenB);

		report.Check(kSuiteTransform, "operator_star_equals_b_transformed_by_a", matchesBThenA,
			"(a*b).TransformPosition(x) should equal a.TransformPosition(b.TransformPosition(x)) -- "
			"i.e. b is treated as being expressed in a's local space (matches glm's quat/mat4 M*v convention); "
			"got (a*b)->" + ToString(viaComposed) + ", a(b(x))->" + ToString(bThenA) + ", b(a(x))->" + ToString(aThenB));

		// document, don't silently pass either way: exactly one reading should match
		report.Check(kSuiteTransform, "operator_star_is_unambiguous", matchesBThenA != matchesAThenB,
			"exactly one of the two composition readings should match observed behavior, not both/neither "
			"(both matching would mean a and b commute here, which they should not for these rotations)");
	}

	static void Test_TransformDefaultForwardIsMinusZ(TestReport& report)
	{
		Transform t; // identity rotation
		float3 forward = t.GetForward();
		float3 expected(0, 0, -1);

		report.Check(kSuiteTransform, "default_forward_is_minus_z", NearlyEqual(forward, expected),
			"Transform::GetForward() with identity rotation must be " + ToString(expected) + " (rot * (0,0,-1)), got " + ToString(forward));
	}

	static void Test_TransformObjectMatrixMatchesTransformPosition(TestReport& report)
	{
		// GetObjectMatrix() and TransformPosition() are two independent code paths that
		// must describe the same transform for a Transform with no scale.
		Transform t;
		t.SetPos(float3(2, -1, 5));
		t.SetRotation(float2(glm::radians(45.0f), glm::radians(15.0f)));

		float3 x(1, 0, 0);
		float3 viaHelper = t.TransformPosition(x);
		float3 viaMatrix = float3(t.GetObjectMatrix() * float4(x, 1.0f));

		report.Check(kSuiteTransform, "object_matrix_matches_transform_position", NearlyEqual(viaHelper, viaMatrix),
			"TransformPosition(x) and GetObjectMatrix()*vec4(x,1) must agree for an unscaled transform, got "
			+ ToString(viaHelper) + " vs " + ToString(viaMatrix));
	}

	//---- GizmoExtraction -------------------------------------------------------------------
	// App.cpp's ImGuizmo integration reads the manipulated matrix back into
	// Transform's position/rot fields per-operation (TRANSLATE only touches position, ROTATE
	// only touches rot -- see drawFrame()'s "Viewport" block) instead of decomposing the whole
	// matrix every frame, after an earlier version (glm::decompose() + an unnecessary
	// glm::conjugate()) made translating a rotated object visibly re-rotate it. These tests
	// pin down the two mathematical facts that fix relies on, independent of ImGuizmo's own
	// (row-vector) matrix convention -- see Renderer.cpp/App.cpp's own comments for
	// why the convention difference doesn't actually matter for a *single* matrix ImGuizmo
	// hands back, only if code tried to manually recompose it with a GLM-multiplied matrix.
	static const char* kSuiteGizmo = "GizmoExtraction";

	// Extracts a Transform's rotation back out of its object matrix's upper-left 3x3 the same
	// way App.cpp's ROTATE branch does: normalize each column to discard whatever
	// scale is baked in (GetObjectMatrix() applies scale before rotation, see Transform.h), then
	// quat_cast the resulting pure rotation matrix. No Euler angles, no glm::decompose() --
	// avoids both the gimbal-lock class of issues and glm::decompose()'s general (for a T*S*R
	// matrix, ambiguous-order) polar decomposition.
	static glm::quat ExtractRotationLikeGizmoRotateBranch(const glm::mat4& model)
	{
		const glm::mat3 rot3(glm::normalize(float3(model[0])), glm::normalize(float3(model[1])), glm::normalize(float3(model[2])));
		return glm::quat_cast(rot3);
	}

	static void Test_TranslatingPreservesRotationBlockExactly(TestReport& report)
	{
		// The fact App.cpp's TRANSLATE branch relies on: for ANY matrix A and a pure
		// translation matrix Tt (identity linear part, offset t), A * Tt's upper-left 3x3 is
		// exactly A's upper-left 3x3 -- true by the block-multiply identity
		// [Ra 0; ta 1] * [I 0; t 1] = [Ra, Ra*t+ta; 0, 1] regardless of what t is, so this holds
		// in ANY row/column-vector convention consistently applied, not just GLM's. This is what
		// ImGuizmo's translate operation actually computes internally (result = modelSource *
		// deltaMatrixTranslation) -- so the manipulated matrix's rotation part literally cannot
		// change from a translate, structurally, before any extraction code even runs.
		Transform t;
		t.SetPos(float3(-0.7f, 2.6f, 0.1f));
		t.SetRotation(float2(1.234f, 0.2f)); // arbitrary yaw+pitch, like a ring-arranged couch

		const glm::mat4 model = t.GetObjectMatrix();
		const glm::mat4 pureTranslation = glm::translate(glm::mat4(1.0f), float3(3.0f, -5.0f, 8.0f));
		const glm::mat4 translated = model * pureTranslation;

		for (int col = 0; col < 3; ++col)
			report.Check(kSuiteGizmo, "translate_preserves_rotation_column_" + std::to_string(col),
				NearlyEqual(float3(translated[col]), float3(model[col])),
				"column " + std::to_string(col) + " of model*translate(t) must equal model's own column "
				+ std::to_string(col) + " (rotation/scale untouched by a pure translation), got "
				+ ToString(float3(translated[col])) + " vs " + ToString(float3(model[col])));

		const glm::quat recovered = ExtractRotationLikeGizmoRotateBranch(translated);
		report.Check(kSuiteGizmo, "translate_then_extract_matches_original_rotation", NearlyEqual(recovered, t.rot),
			"extracting rotation from a translated-but-not-rotated matrix must still match the original, got "
			+ ToString(recovered) + " vs " + ToString(t.rot));
	}

	static void Test_RotateExtractionRecoversKnownRotation(TestReport& report)
	{
		// A handful of rotations, including ones near where a naive Euler-angle round-trip
		// (e.g. ImGuizmo's own DecomposeMatrixToComponents/RecomposeMatrixFromComponents, or
		// glm::decompose()'s polar decomposition) tends to show its seams -- quat_cast of a
		// normalized rotation matrix has no such singularities, so all of these should recover
		// exactly.
		const glm::quat cases[] = {
			glm::quat(1, 0, 0, 0),
			glm::angleAxis(glm::radians(37.0f), glm::normalize(float3(0.2f, 1.0f, 0.3f))),
			glm::angleAxis(glm::radians(90.0f), float3(0, 1, 0)),
			glm::angleAxis(glm::radians(179.0f), float3(0, 1, 0)), // near the 180-degree case
			glm::angleAxis(glm::radians(-125.0f), glm::normalize(float3(1, 1, 1))),
		};

		for (size_t i = 0; i < std::size(cases); ++i)
		{
			Transform t;
			t.SetPos(float3(2, -1, 5));
			t.rot = cases[i];

			const glm::quat recovered = ExtractRotationLikeGizmoRotateBranch(t.GetObjectMatrix());
			report.Check(kSuiteGizmo, "rotate_extraction_case_" + std::to_string(i), NearlyEqual(recovered, cases[i]),
				"case " + std::to_string(i) + ": expected " + ToString(cases[i]) + ", got " + ToString(recovered));
		}
	}

	static void Test_RepeatedRotateExtractionDoesNotDrift(TestReport& report)
	{
		// Simulates the part of a live drag my extraction code is actually responsible for:
		// each frame, ImGuizmo hands back *some* new rotation matrix (folding in this frame's
		// mouse-driven delta -- represented here by composing a small known extra rotation, since
		// ImGuizmo's own math is proven convention-safe for a single returned matrix, see this
		// suite's header comment), the code extracts a quat from it and writes Transform::rot,
		// and next frame that quat is fed back through GetObjectMatrix() into the next
		// iteration. Round-tripping through quat_cast(normalize(columns)) a hundred times must
		// not accumulate any drift, unlike a general decompose()-based approach.
		glm::quat rot = glm::angleAxis(glm::radians(12.0f), glm::normalize(float3(0.1f, 1.0f, 0.05f)));
		const glm::quat perFrameDelta = glm::angleAxis(glm::radians(0.7f), float3(0, 1, 0));

		glm::quat expected = rot;
		const int kFrames = 200;
		for (int frame = 0; frame < kFrames; ++frame)
		{
			expected = glm::normalize(perFrameDelta * expected);

			Transform t;
			t.rot = rot;
			glm::mat4 model = t.GetObjectMatrix();
			// Fold in "this frame's" incremental rotation the same way applying perFrameDelta to
			// the object's world orientation would -- mirrors composing a rotation delta onto an
			// existing quat, not a fresh angle-axis construction, since that's what would let
			// per-frame floating point error compound if it were going to.
			model = glm::mat4(perFrameDelta) * model;

			rot = ExtractRotationLikeGizmoRotateBranch(model);
		}

		report.Check(kSuiteGizmo, "repeated_rotate_extraction_matches_expected_after_200_frames", NearlyEqual(rot, expected),
			"200 frames of extract/rebuild/re-rotate must land on the same orientation as composing the deltas "
			"directly, no drift; expected " + ToString(expected) + ", got " + ToString(rot));

		float3 probe(0, 0, -1);
		report.Check(kSuiteGizmo, "repeated_rotate_extraction_vector_matches", NearlyEqual(expected * probe, rot * probe),
			"sanity check via a rotated probe vector (avoids only comparing quats that could both be equally-wrong), "
			"expected " + ToString(expected * probe) + ", got " + ToString(rot * probe));
	}

	void RunGlmMathTests(TestReport& report)
	{
		Test_IdentityIsNoOp(report);
		Test_TranslationMovesPointsNotDirections(report);
		Test_MatrixMultiplyOrderMatters(report);
		Test_MultiplyIsAssociative(report);
		Test_ColumnMajorStorage(report);
		Test_QuatMatrixAgree(report);

		Test_QuatConstructorArgumentOrderIsWXYZ(report);
		Test_QuatDoubleCover(report);
		Test_QuatCompositionOrder(report);

		Test_TransformCompositionMatchesSequentialApplication(report);
		Test_TransformDefaultForwardIsMinusZ(report);
		Test_TransformObjectMatrixMatchesTransformPosition(report);

		Test_TranslatingPreservesRotationBlockExactly(report);
		Test_RotateExtractionRecoversKnownRotation(report);
		Test_RepeatedRotateExtractionDoesNotDrift(report);
	}
}
