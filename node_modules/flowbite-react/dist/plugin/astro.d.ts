declare const _default: () => {
    name: string;
    hooks: {
        "astro:config:setup": (astro: any) => Promise<void>;
    };
};
export default _default;
