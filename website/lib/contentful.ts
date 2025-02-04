import { createClient } from "contentful";

export const client = createClient({
  space: process.env.CONTENTFUL_SPACE_ID!,
  accessToken: process.env.CONTENTFUL_ACCESS_TOKEN!,
});

export interface ImageEntry {
  sys: {
    id: string;
  };
  fields: {
    image: {
      fields: {
        file: {
          url: string;
        };
      };
    };
  };
}

export interface AicaEntry {
  sys: {
    id: string;
  };
  fields: {
    image: {
      fields: {
        file: {
          url: string;
        };
      };
    };
  };
}
