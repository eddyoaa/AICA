"use client";

export default function Imprint() {
  return (
    <main className="min-h-screen bg-black text-white p-8">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-4xl font-bold mb-8">Imprint</h1>
        
        <div className="space-y-6 text-gray-300">
          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">Information according to § 5 TMG</h2>
            <p>AI&apos;m Sitting in a Room</p>
            <p>Sample Street 123</p>
            <p>12345 Sample City</p>
            <p>Germany</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">Contact</h2>
            <p>Phone: +49 (0) 123 456789</p>
            <p>Email: contact@example.com</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">Responsible for Content</h2>
            <p>John Doe</p>
            <p>Sample Street 123</p>
            <p>12345 Sample City</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">Dispute Resolution</h2>
            <p>The European Commission provides a platform for online dispute resolution (OS): https://ec.europa.eu/consumers/odr/</p>
            <p>We are not willing or obliged to participate in dispute resolution proceedings before a consumer arbitration board.</p>
          </section>
        </div>
      </div>
    </main>
  );
}