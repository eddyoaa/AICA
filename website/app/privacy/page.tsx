"use client";

export default function Privacy() {
  return (
    <main className="min-h-screen bg-black text-white p-8">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-4xl font-bold mb-8">Privacy Policy</h1>
        
        <div className="space-y-6 text-gray-300">
          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">1. Data Protection Overview</h2>
            <p>We take the protection of your personal data very seriously. This privacy policy informs you about how we handle your personal data when you visit our website.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">2. Data Collection on Our Website</h2>
            <h3 className="text-xl font-semibold text-white mb-2">Server Log Files</h3>
            <p>The website provider automatically collects and stores information in server log files that your browser automatically transmits to us. These are:</p>
            <ul className="list-disc list-inside mt-2 ml-4">
              <li>Browser type and version</li>
              <li>Operating system used</li>
              <li>Referrer URL</li>
              <li>Hostname of the accessing computer</li>
              <li>Time of the server request</li>
              <li>IP address</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">3. Image Downloads</h2>
            <p>When you download images from our website, we collect the following data:</p>
            <ul className="list-disc list-inside mt-2 ml-4">
              <li>Time of download</li>
              <li>Downloaded image information</li>
              <li>IP address</li>
            </ul>
            <p className="mt-2">This data is collected to ensure the security and stability of our service.</p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">4. Your Rights</h2>
            <p>You have the following rights regarding your personal data:</p>
            <ul className="list-disc list-inside mt-2 ml-4">
              <li>Right to access</li>
              <li>Right to rectification</li>
              <li>Right to erasure</li>
              <li>Right to restriction of processing</li>
              <li>Right to data portability</li>
              <li>Right to object</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">5. Contact</h2>
            <p>If you have any questions about this privacy policy, please contact us at:</p>
            <p>Email: privacy@example.com</p>
            <p>Phone: +49 (0) 123 456789</p>
          </section>
        </div>
      </div>
    </main>
  );
}