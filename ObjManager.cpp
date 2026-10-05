#include "global.h"
#include "ObjManager.h"

void CObjManagerTest()
{

	{
		CObjManager<CObjHandle, char, 2> man;

		ASSERT(man.GetElementCount() == 0);
		ASSERT(man.GetDeadCount() == 0);

		man.InitOnce(3);

		CObjHandle a, b, c, d, e, f, g, aa;

		ASSERT(a == b);
		ASSERT(!(a != b));
		ASSERT(a == a);
		ASSERT(!(a != a));

		ASSERT(man.GetElementCount() == 0);

		a = man.push_back('a');
		ASSERT(a.IsValid());
		ASSERT(a.Index == 0 && a.Seed == 0);
		ASSERT(*man.GetElementPtr(a) == 'a');
		ASSERT(man.GetElementCount() == 1);
		ASSERT(man.GetElementByIterator(0) == 'a');

		aa = a;
		ASSERT(a == aa);
		ASSERT(aa == a);
		ASSERT(!(a != aa));
		ASSERT(a != b);

		b = man.push_back('b');
		ASSERT(b.IsValid());
		ASSERT(b.Index == 1 && b.Seed == 0);
		ASSERT(*man.GetElementPtr(b) == 'b');
		ASSERT(man.GetElementCount() == 2);
		ASSERT(man.GetElementByIterator(0) == 'a');
		ASSERT(man.GetElementByIterator(1) == 'b');

		c = man.push_back('c');
		ASSERT(c.IsValid());
		ASSERT(c.Index == 2 && c.Seed == 0);
		ASSERT(*man.GetElementPtr(c) == 'c');
		ASSERT(man.GetElementCount() == 3);
		ASSERT(man.GetElementByIterator(0) == 'a');
		ASSERT(man.GetElementByIterator(1) == 'b');
		ASSERT(man.GetElementByIterator(2) == 'c');

		{
			CObjContainer<CObjHandle, 2> cont;
			uint32 index;

			cont.InitOnce();
			ASSERT(cont.GetCount() == 0);
			ASSERT(cont.Find(b) == -1);

			cont.AddOrUpdate(b);
			ASSERT(cont.GetCount() == 1);
			index = 0;
			ASSERT(cont.Iterate(index) == b);
			ASSERT(!cont.Iterate(index).IsValid());
			ASSERT(cont.Find(b) != -1);

			cont.AddOrUpdate(c);
			index = 0;
			ASSERT(cont.GetCount() == 2);
			ASSERT(cont.Iterate(index) == b);
			ASSERT(cont.Iterate(index) == c);
			ASSERT(!cont.Iterate(index).IsValid());
			ASSERT(cont.Find(b) != -1);
			ASSERT(cont.Find(c) != -1);

			cont.AddOrUpdate(c);
			index = 0;
			ASSERT(cont.GetCount() == 2);
			ASSERT(cont.Iterate(index) == b);
			ASSERT(cont.Iterate(index) == c);
			ASSERT(!cont.Iterate(index).IsValid());
			ASSERT(cont.Find(b) != -1);
			ASSERT(cont.Find(c) != -1);

			index = 0;
			cont.AddOrUpdate(a);
			ASSERT(cont.GetCount() == 3);
			ASSERT(cont.Iterate(index) == b);
			ASSERT(cont.Iterate(index) == c);
			ASSERT(cont.Iterate(index) == a);
			ASSERT(!cont.Iterate(index).IsValid());
			ASSERT(cont.Find(a) != -1);
			ASSERT(cont.Find(b) != -1);
			ASSERT(cont.Find(c) != -1);

			cont.Remove(b);
			index = 0;
			ASSERT(cont.GetCount() == 2);
			ASSERT(cont.Iterate(index) == a);
			ASSERT(cont.Iterate(index) == c);
			ASSERT(!cont.Iterate(index).IsValid());
			ASSERT(cont.Find(a) != -1);
			ASSERT(cont.Find(b) == -1);
			ASSERT(cont.Find(c) != -1);

			cont.Remove(b);
			index = 0;
			ASSERT(cont.GetCount() == 2);
			ASSERT(cont.Iterate(index) == a);
			ASSERT(cont.Iterate(index) == c);
			ASSERT(!cont.Iterate(index).IsValid());
			ASSERT(cont.Find(a) != -1);
			ASSERT(cont.Find(b) == -1);
			ASSERT(cont.Find(c) != -1);

			cont.Remove(c);
			index = 0;
			ASSERT(cont.GetCount() == 1);
			ASSERT(cont.Iterate(index) == a);
			ASSERT(!cont.Iterate(index).IsValid());
			ASSERT(cont.Find(a) != -1);
			ASSERT(cont.Find(b) == -1);
			ASSERT(cont.Find(c) == -1);

			cont.Remove(c);
			index = 0;
			ASSERT(cont.GetCount() == 1);
			ASSERT(cont.Iterate(index) == a);
			ASSERT(!cont.Iterate(index).IsValid());
			ASSERT(cont.Find(a) != -1);
			ASSERT(cont.Find(b) == -1);
			ASSERT(cont.Find(c) == -1);

			cont.Remove(a);
			index = 0;
			ASSERT(cont.GetCount() == 0);
			ASSERT(!cont.Iterate(index).IsValid());
			ASSERT(cont.Find(a) == -1);
			ASSERT(cont.Find(b) == -1);
			ASSERT(cont.Find(c) == -1);
		}

		d = man.push_back('d');
		ASSERT(!d.IsValid());
		ASSERT(!man.GetElementPtr(d));
		ASSERT(man.GetElementCount() == 3);

		man.Remove(b);
		ASSERT(b.IsValid());
		ASSERT(!man.GetElementPtr(b));
		ASSERT(man.GetElementCount() == 2);
		ASSERT(man.GetDeadCount() == 0);
		ASSERT(man.GetElementByIterator(0) == 'a');
		ASSERT(man.GetElementByIterator(1) == 'c');

		e = man.push_back('e');
		ASSERT(e.IsValid());
		ASSERT(e.Index == 1 && e.Seed == 1);
		ASSERT(*man.GetElementPtr(e) == 'e');
		ASSERT(!man.GetElementPtr(b));
		ASSERT(man.GetElementCount() == 3);
		ASSERT(man.GetElementByIterator(0) == 'a');
		ASSERT(man.GetElementByIterator(1) == 'c');
		ASSERT(man.GetElementByIterator(2) == 'e');

		man.Remove(e);
		ASSERT(e.IsValid());
		ASSERT(!man.GetElementPtr(e));
		ASSERT(man.GetElementCount() == 2);
		ASSERT(man.GetDeadCount() == 1);

		f = man.push_back('f');
		ASSERT(!f.IsValid());
		ASSERT(!man.GetElementPtr(f));
		ASSERT(man.GetElementCount() == 2);
		ASSERT(man.GetElementByIterator(0) == 'a');
		ASSERT(man.GetElementByIterator(1) == 'c');

		man.Remove(a);
		ASSERT(a.IsValid());
		ASSERT(!man.GetElementPtr(a));
		ASSERT(man.GetElementByIterator(0) == 'c');
		ASSERT(man.GetElementCount() == 1);
		ASSERT(man.GetDeadCount() == 1);

		man.Remove(c);
		ASSERT(c.IsValid());
		ASSERT(!man.GetElementPtr(c));
		ASSERT(man.GetElementCount() == 0);
		ASSERT(man.GetDeadCount() == 1);

		g = man.push_back('g');
		ASSERT(g.IsValid());
		ASSERT(*man.GetElementPtr(g) == 'g');
		ASSERT(man.GetElementCount() == 1);
		ASSERT(man.GetElementByIterator(0) == 'g');

		man.Remove(g);
		ASSERT(g.IsValid());
		ASSERT(!man.GetElementPtr(g));
		ASSERT(man.GetElementCount() == 0);
		ASSERT(man.GetDeadCount() == 2);
	}

	struct SHndExtra : public CObjHandle
	{
		SHndExtra(const CObjHandle& In)
			: CObjHandle(In)
			, Extra(0)
		{
		}

		SHndExtra(int InExtra = 0)
			: Extra(InExtra)
		{
		}

		// comparison operator
		bool operator==(const SHndExtra& rhs) const
		{
			return (CObjHandle&)*this == (CObjHandle&)rhs && Extra == rhs.Extra;
		}

		int Extra;
	};

	{
		CObjManager<CObjHandle, char, 2> man;

		ASSERT(man.GetElementCount() == 0);
		ASSERT(man.GetDeadCount() == 0);

		man.InitOnce(3);

		SHndExtra a, b, c;

		a = man.push_back('a');
		a.Extra = 1;
		b = man.push_back('b');
		b.Extra = 2;
		c = man.push_back('c');
		c.Extra = 3;

		CObjContainer<SHndExtra> cont;
		uint32 index;

		cont.InitOnce();
		ASSERT(cont.GetCount() == 0);
		ASSERT(cont.Find(b) == -1);

		cont.AddOrUpdate(b);
		ASSERT(cont.GetCount() == 1);
		index = 0;
		ASSERT(cont.Iterate(index) == b);
		ASSERT(!cont.Iterate(index).IsValid());
		ASSERT(cont.Find(b) != -1);

		cont.AddOrUpdate(c);
		index = 0;
		ASSERT(cont.GetCount() == 2);
		ASSERT(cont.Iterate(index) == b);
		ASSERT(cont.Iterate(index) == c);
		ASSERT(!cont.Iterate(index).IsValid());
		ASSERT(cont.Find(b) != -1);
		ASSERT(cont.Find(c) != -1);

		cont.AddOrUpdate(c);
		index = 0;
		ASSERT(cont.GetCount() == 2);
		ASSERT(cont.Iterate(index) == b);
		ASSERT(cont.Iterate(index) == c);
		ASSERT(!cont.Iterate(index).IsValid());
		ASSERT(cont.Find(b) != -1);
		ASSERT(cont.Find(c) != -1);

		index = 0;
		cont.AddOrUpdate(a);
		ASSERT(cont.GetCount() == 3);
		ASSERT(cont.Iterate(index) == b);
		ASSERT(cont.Iterate(index) == c);
		ASSERT(cont.Iterate(index) == a);
		ASSERT(!cont.Iterate(index).IsValid());
		ASSERT(cont.Find(a) != -1);
		ASSERT(cont.Find(b) != -1);
		ASSERT(cont.Find(c) != -1);

		cont.Remove(b);
		index = 0;
		ASSERT(cont.GetCount() == 2);
		ASSERT(cont.Iterate(index) == a);
		ASSERT(cont.Iterate(index) == c);
		ASSERT(!cont.Iterate(index).IsValid());
		ASSERT(cont.Find(a) != -1);
		ASSERT(cont.Find(b) == -1);
		ASSERT(cont.Find(c) != -1);

		cont.Remove(b);
		index = 0;
		ASSERT(cont.GetCount() == 2);
		ASSERT(cont.Iterate(index) == a);
		ASSERT(cont.Iterate(index) == c);
		ASSERT(!cont.Iterate(index).IsValid());
		ASSERT(cont.Find(a) != -1);
		ASSERT(cont.Find(b) == -1);
		ASSERT(cont.Find(c) != -1);

		cont.Remove(c);
		index = 0;
		ASSERT(cont.GetCount() == 1);
		ASSERT(cont.Iterate(index) == a);
		ASSERT(!cont.Iterate(index).IsValid());
		ASSERT(cont.Find(a) != -1);
		ASSERT(cont.Find(b) == -1);
		ASSERT(cont.Find(c) == -1);

		cont.Remove(c);
		index = 0;
		ASSERT(cont.GetCount() == 1);
		ASSERT(cont.Iterate(index) == a);
		ASSERT(!cont.Iterate(index).IsValid());
		ASSERT(cont.Find(a) != -1);
		ASSERT(cont.Find(b) == -1);
		ASSERT(cont.Find(c) == -1);

		cont.Remove(a);
		index = 0;
		ASSERT(cont.GetCount() == 0);
		ASSERT(!cont.Iterate(index).IsValid());
		ASSERT(cont.Find(a) == -1);
		ASSERT(cont.Find(b) == -1);
		ASSERT(cont.Find(c) == -1);
	}

}
