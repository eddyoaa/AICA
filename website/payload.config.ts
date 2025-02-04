import { buildConfig } from 'payload/config';
import { webpackBundler } from '@payloadcms/bundler-webpack';
import { mongooseAdapter } from '@payloadcms/db-mongodb';
import { slateEditor } from '@payloadcms/richtext-slate';
import path from 'path';

export default buildConfig({
  serverURL: process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:3000',
  admin: {
    bundler: webpackBundler(),
  },
  editor: slateEditor({}),
  collections: [
    {
      slug: 'images',
      admin: {
        useAsTitle: 'title',
      },
      access: {
        read: () => true,
      },
      fields: [
        {
          name: 'title',
          type: 'text',
          required: true,
        },
        {
          name: 'image',
          type: 'upload',
          relationTo: 'media',
          required: true,
        },
        {
          name: 'description',
          type: 'textarea',
        },
        {
          name: 'position',
          type: 'number',
          required: true,
        }
      ],
    },
    {
      slug: 'media',
      upload: {
        staticURL: '/media',
        staticDir: 'media',
        imageSizes: [
          {
            name: 'thumbnail',
            width: 400,
            height: 300,
            position: 'centre',
          },
          {
            name: 'card',
            width: 768,
            height: 1024,
            position: 'centre',
          }
        ],
        mimeTypes: ['image/*'],
      },
    },
  ],
  db: mongooseAdapter({
    url: process.env.MONGODB_URI || 'mongodb://localhost/payload-gallery',
  }),
  typescript: {
    outputFile: path.resolve(__dirname, 'payload-types.ts'),
  },
});